from io import BytesIO                   # pozwala nam "udawać" plik audio w pamięci
import streamlit as st                  # biblioteka do tworzenia prostych aplikacji internetowych
from audiorecorder import audiorecorder  # do nagrywania dźwięku z mikrofonu
from dotenv import dotenv_values          # do odczytania tajnych danych z pliku .env
from hashlib import md5                   # tworzy „odcisk palca” pliku (do sprawdzenia, czy się zmienił)
from openai import OpenAI                 # do korzystania z usług OpenAI (np. rozpoznawania mowy)
from qdrant_client import QdrantClient    # baza danych do przechowywania notatek
from qdrant_client.models import PointStruct, Distance, VectorParams  # elementy potrzebne do przechowywania danych wektorowych

# """
#     conda remove pydub
#     conda install -c conda-forge pydub

#     conda install -c conda-forge ffmpeg=6.1.1
# """

# 
# conda activate app_notatki

env = dotenv_values(".env")
### Secrets using Streamlit Cloud Mechanism
# https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
if 'QDRANT_URL' in st.secrets:
    env['QDRANT_URL'] = st.secrets['QDRANT_URL']
if 'QDRANT_API_KEY' in st.secrets:
    env['QDRANT_API_KEY'] = st.secrets['QDRANT_API_KEY']
###


# Ustawiamy nazwy i liczby potrzebne do działania
EMBEDDING_MODEL = "text-embedding-3-large"     # model do zamiany tekstu na liczby (wektory)
EMBEDDING_DIM = 3072                           # ile liczb ma mieć jeden wektor
AUDIO_TRANSCRIBE_MODEL = "whisper-1"           # model do zamiany mowy na tekst
QDRANT_COLLECTION_NAME = "notes"               # nazwa kolekcji (czyli folderu) na notatki

# Funkcja tworzy klienta OpenAI z kluczem API
def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])

# Funkcja zmienia nagranie audio na tekst
def transcribe_audio(audio_bytes):
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)      # zamieniamy dane audio na "plik"
    audio_file.name = "audio.mp3"          # nadajemy mu nazwę
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIBE_MODEL,
        response_format="verbose_json",    # format odpowiedzi
    )
    return transcript.text                # zwracamy tekst

# Tworzymy klienta bazy danych (Qdrant)
@st.cache_resource
def get_qdrant_client():
    # return QdrantClient(path=":memory:")   # baza działa w pamięci (nie zapisuje nic na dysku)
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
)


# Sprawdzamy, czy kolekcja notatek istnieje — jeśli nie, tworzymy ją
def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,  # jak liczymy podobieństwo między notatkami
            ),
        )
    else:
        print("Kolekcja już istnieje")

# Funkcja zamienia tekst na wektor (czyli listę liczb)
def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding

# Funkcja dodaje notatkę (tekst) do bazy danych
def add_note_to_db(note_text):
    qdrant_client = get_qdrant_client()
    points_count = qdrant_client.count(
        collection_name=QDRANT_COLLECTION_NAME,
        exact=True,
    )
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION_NAME,
        points=[
            PointStruct(
                id=points_count.count + 1,                   # unikalny numer notatki
                vector=get_embedding(text=note_text),        # wektor tekstu
                payload={"text": note_text},                 # zapisany tekst
            )
        ]
    )

# Funkcja pobiera notatki z bazy (z opcjonalnym wyszukiwaniem)
def list_notes_from_db(query=None):
    qdrant_client = get_qdrant_client()
    if not query:
        # Jeśli nie ma zapytania, zwróć 10 pierwszych notatek
        notes = qdrant_client.scroll(collection_name=QDRANT_COLLECTION_NAME, limit=10)[0]
        result = []
        for note in notes:
            result.append({"text": note.payload["text"], "score": None})
        return result
    else:
        # Jeśli podano zapytanie, znajdź najbardziej podobne notatki
        notes = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=get_embedding(text=query),
            limit=10,
        )
        result = []
        for note in notes:
            result.append({
                "text": note.payload["text"],
                "score": note.score,
            })
        return result

# Ustawienia wyglądu strony
st.set_page_config(page_title="Audio Notatki", layout="centered")

# Sprawdzamy, czy mamy klucz API — jeśli nie, prosimy użytkownika o podanie
if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAI aby móc korzystać z tej aplikacji")
        st.session_state["openai_api_key"] = st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

# Przygotowanie pamięci na dane w sesji
if "note_audio_bytes_md5" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None
if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None
if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""
if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

# Nagłówek aplikacji
st.title("Audio Notatki")
assure_db_collection_exists()  # upewniamy się, że kolekcja w bazie istnieje

# Tworzymy dwie zakładki: dodawanie i wyszukiwanie notatek
add_tab, search_tab = st.tabs(["Dodaj notatkę", "Wyszukaj notatkę"])

# Zakładka: Dodawanie notatki
with add_tab:
    note_audio = audiorecorder(   # przycisk do nagrywania audio
        start_prompt="Nagraj notatkę",
        stop_prompt="Zatrzymaj nagrywanie",
    )
    if note_audio:
        audio = BytesIO()                          # tworzymy plik audio w pamięci
        note_audio.export(audio, format="mp3")     # zapisujemy nagranie jako mp3
        st.session_state["note_audio_bytes"] = audio.getvalue()
        current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()

        # Jeśli nagranie się zmieniło — resetujemy tekst
        if st.session_state["note_audio_bytes_md5"] != current_md5:
            st.session_state["note_audio_text"] = ""
            st.session_state["note_text"] = ""
            st.session_state["note_audio_bytes_md5"] = current_md5

        st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")  # odtwarzanie nagrania

        # Przycisk do transkrypcji
        if st.button("Transkrybuj audio"):
            st.session_state["note_audio_text"] = transcribe_audio(st.session_state["note_audio_bytes"])

        # Jeśli mamy transkrypcję, pokazujemy ją do edycji
        if st.session_state["note_audio_text"]:
            st.session_state["note_text"] = st.text_area("Edytuj notatkę", value=st.session_state["note_audio_text"])

        # Zapisz notatkę do bazy, jeśli coś wpisano
        if st.session_state["note_text"] and st.button("Zapisz notatkę", disabled=not st.session_state["note_text"]):
            add_note_to_db(note_text=st.session_state["note_text"])
            st.toast("Notatka zapisana", icon="🎉")

# Zakładka: Wyszukiwanie notatek
with search_tab:
    query = st.text_input("Wyszukaj notatkę")  # użytkownik wpisuje zapytanie
    if st.button("Szukaj"):                    # po kliknięciu "Szukaj"
        for note in list_notes_from_db(query):
            with st.container(border=True):   # każda notatka w ramce
                st.markdown(note["text"])     # wyświetlamy tekst notatki
                if note["score"]:             # jeśli podano trafność, pokazujemy ją
                    st.markdown(f':violet[{note["score"]}]')
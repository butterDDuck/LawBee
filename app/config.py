"""환경 설정 로딩 (.env 에서 값 읽어옴)"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_vision_model: str = "gpt-4o-mini"
    openai_stt_model: str = "whisper-1"

    data_path: str = "data/regulations.jsonl"
    vectorstore_dir: str = "vectorstore"
    db_path: str = "lawbee.db"
    uploads_dir: str = "var/uploads"


settings = Settings()

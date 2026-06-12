"""환경 설정 로딩 (.env 에서 값 읽어옴)"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    data_path: str = "data/regulations.jsonl"
    laws_dir: str = "doc"
    vectorstore_dir: str = "vectorstore"
    db_path: str = "lawbee.db"


settings = Settings()

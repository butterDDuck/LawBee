"""환경 설정 로딩 (.env 에서 값 읽어옴)"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-small"
    openai_judge_model: str = "gpt-4.1"     # 심의·재심의 판단 (정확도 우선)
    openai_chat_model: str = "gpt-4o-mini"  # 대안 문구 생성 (비용 우선)
    openai_vision_model: str = "gpt-4o"     # 화면·이미지 분석 (가독성 판독 우선)
    openai_stt_model: str = "whisper-1"

    data_path: str = "data/regulations_merged.jsonl"
    laws_dir: str = "doc"
    vectorstore_dir: str = "vectorstore"
    db_path: str = "lawbee.db"
    uploads_dir: str = "var/uploads"


settings = Settings()

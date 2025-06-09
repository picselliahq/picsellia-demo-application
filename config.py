from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class PicselliaConfig(BaseSettings):
    """Configuration settings for Picsellia integration."""
    
    PICSELLIA_TOKEN: str = Field(
        default=os.getenv("PICSELLIA_TOKEN", ""),
        description="Picsellia API token for authentication"
    )

    PICSELLIA_ORGANIZATION: str = Field(
        default=os.getenv("PICSELLIA_ORGANIZATION", ""),
        description="Picsellia organization token for authentication"
    )

    COCO_DEPLOYMENT_ID: str = Field(
        default=os.getenv("COCO_DEPLOYMENT_ID", ""),
        description="COCO Picsellia deployment ID"
    )

    FACE_DEPLOYMENT_ID: str = Field(
        default=os.getenv("FACE_DEPLOYMENT_ID", ""),
        description="Face Picsellia deployment ID"
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True

# Create a global config instance
config = PicselliaConfig()

# Validate configuration on import
if not config.PICSELLIA_TOKEN:
    print("Warning: PICSELLIA_TOKEN is not set. Please set it in your environment or .env file.")

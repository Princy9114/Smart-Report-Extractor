from typing import Optional
from pydantic import BaseModel
from .base import BaseExtraction


class WorkExperience(BaseModel):
    company: Optional[str] = None
    title: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: Optional[str] = None


class Education(BaseModel):
    institution: Optional[str] = None
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    year: Optional[str] = None


class ResumeExtraction(BaseExtraction):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    skills: list[str] = []
    experience: list[WorkExperience] = []
    education: list[Education] = []
    certifications: list[str] = []
    languages: list[str] = []
    total_years_experience: Optional[float] = None

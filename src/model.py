from typing import List, Optional
from pydantic import BaseModel

class HyperLink(BaseModel):
    url: str
    text: str

class ContactInfo(BaseModel):
    name: str
    phone: str
    location: str
    email: HyperLink
    website: HyperLink
    linkedin: HyperLink

class Objective(BaseModel):
    text: str

class EducationItem(BaseModel):
    university: str
    degree: str
    major: str
    graduation_date: str
    coursework: List[str]
    gpa: float

class Skills(BaseModel):
    languages_databases: List[str]
    cloud: List[str]
    ai_ml: List[str]
    development: List[str]
    others: List[str]

class ExperienceItem(BaseModel):
    company: str
    title: str
    location: str
    dates: str
    description: List[str]

class ProjectItem(BaseModel):
    project_name: str
    technologies: List[str]
    description: List[str]

class Certifications(BaseModel):
    certifications: List[str]


class Resume(BaseModel):
    contact_info: ContactInfo
    objective: Objective
    education: List[EducationItem]
    skills: Skills
    experience: List[ExperienceItem]
    projects: List[ProjectItem]
    certifications: Certifications
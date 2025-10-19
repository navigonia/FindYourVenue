import uuid
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
from .database import Base

class User(Base):
    """
    Benutzer, die sich in der Anwendung anmelden können.
    Jeder Benutzer kann mehrere Lärmpunkte besitzen.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

   
    points = relationship("NoisePoint", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User id={self.id} username={self.username}>"


class NoisePoint(Base):
    
    __tablename__ = "noise_points"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    geom = Column(Geometry(geometry_type="POINT", srid=4326), nullable=False)
    db_value = Column(Integer, nullable=False)
    note = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    user = relationship("User", back_populates="points")

    def __repr__(self):
        return f"<NoisePoint id={self.id} db={self.db_value} user_id={self.user_id}>"

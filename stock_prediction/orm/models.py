from typing import List

from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship

from sqlalchemy import String, Integer, Float, ARRAY, DATE, ForeignKey


class Base(DeclarativeBase):
    pass


class ModelParametersDb(Base):
    __tablename__ = "model_parameters"

    id: Mapped[str] = mapped_column(String, unique=True, primary_key=True)
    ticker: Mapped[str] = mapped_column(String(3))
    model: Mapped[str] = mapped_column(String(10))
    units: Mapped[int] = mapped_column(Integer)
    input_shape_min: Mapped[int] = mapped_column(Integer)
    input_shape_max: Mapped[int] = mapped_column(Integer)
    epochs: Mapped[int] = mapped_column(Integer())
    batch_size: Mapped[int] = mapped_column(Integer)
    activation: Mapped[str] = mapped_column(String(10))
    optimizer: Mapped[str] = mapped_column(String(10))

    info: Mapped[List["ModelInfoDb"]] = relationship(
        back_populates="params", cascade="all"
    )


class ModelInfoDb(Base):
    __tablename__ = "model_info"

    id: Mapped[str] = mapped_column(ForeignKey("model_parameters.id"), primary_key=True)
    dataset: Mapped[ARRAY[float]] = mapped_column(ARRAY(Float(precision=2)))
    train_predict: Mapped[ARRAY[float]] = mapped_column(ARRAY(Float(precision=2)))
    test_predict: Mapped[ARRAY[float]] = mapped_column(ARRAY(Float(precision=2)))
    mse_train: Mapped[float] = mapped_column(Float(precision=4))
    mse_test: Mapped[float] = mapped_column(Float(precision=4))
    create_date: Mapped[DATE] = mapped_column(DATE())

    params: Mapped[List["ModelParametersDb"]] = relationship(
        back_populates="info", cascade="all"
    )

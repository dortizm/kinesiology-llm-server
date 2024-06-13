#!/usr/bin/env python
"""A server for the chain above."""

from fastapi import FastAPI
from langserve import add_routes
from app.text_generator import text_generator 
from app.barrier import barrier
from app.facilitator import facilitator
from app.text_analysis import text_analysis
from app.bot import bot

app = FastAPI(title="Retrieval App")

add_routes(app, text_generator, path="/text_generator")
add_routes(app, barrier, path="/barrier")
add_routes(app, facilitator, path="/facilitator")
add_routes(app, text_analysis, path="/text_analysis")
add_routes(app, bot, path="/bot")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
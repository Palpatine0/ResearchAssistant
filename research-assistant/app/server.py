from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.web_chain import chain as web_chain
from app.doc_chain import chain as doc_chain

app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, web_chain, path = "/research-assistant-web_chain")
add_routes(app, doc_chain, path = "/research-assistant-doc_chain")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

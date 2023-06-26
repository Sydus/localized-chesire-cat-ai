import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware

from cat.log import log, welcome
from cat.routes import base, memory, plugins, upload, websocket
from cat.routes.static import public, admin, static
from cat.api_auth import check_api_key
from cat.routes.openapi import get_openapi_configuration_function
from cat.routes.setting import llm_setting, general_setting, embedder_setting, prompt_setting
from cat.looking_glass.cheshire_cat import CheshireCat


@asynccontextmanager
async def lifespan(app: FastAPI):
    #       ^._.^
    #
    # loads Cat and plugins
    # Every endpoint can access the cat instance via request.app.state.ccat
    # - Not using midlleware because I can't make it work with both http and websocket;
    # - Not using Depends because it only supports callables (not instances)
    # - Starlette allows this: https://www.starlette.io/applications/#storing-state-on-the-app-instance
    app.state.ccat = CheshireCat()

    # startup message with admin, public and swagger addresses
    welcome()

    yield


# REST API
cheshire_cat_api = FastAPI(lifespan=lifespan, dependencies=[Depends(check_api_key)])

# Configures the CORS middleware for the FastAPI app
cors_allowed_origins_str = os.getenv("CORS_ALLOWED_ORIGINS", "")
origins = cors_allowed_origins_str.split(",") if cors_allowed_origins_str else ["*"]
cheshire_cat_api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add routers to the middleware stack.
cheshire_cat_api.include_router(base.router, tags=["Status"])
cheshire_cat_api.include_router(general_setting.router, tags=["Settings - General"], prefix="/settings")
cheshire_cat_api.include_router(prompt_setting.router, tags=["Settings - Prompt"], prefix="/settings/prompt")
cheshire_cat_api.include_router(llm_setting.router, tags=["Settings - Large Language Model"], prefix="/settings/llm")
cheshire_cat_api.include_router(embedder_setting.router, tags=["Settings - Embedder"], prefix="/settings/embedder")
cheshire_cat_api.include_router(plugins.router, tags=["Plugins"], prefix="/plugins")
cheshire_cat_api.include_router(memory.router, tags=["Memory"], prefix="/memory")
cheshire_cat_api.include_router(upload.router, tags=["Rabbit Hole (file upload)"], prefix="/rabbithole")
cheshire_cat_api.include_router(websocket.router, tags=["Websocket"])


# mount static files
# this cannot be done via fastapi.APIrouter:
# https://github.com/tiangolo/fastapi/discussions/9070



# admin (index file with dynamic configuration)
admin.mount_admin_index(cheshire_cat_api)
# admin (static build)
admin.mount(cheshire_cat_api)
# static files (for plugins and other purposes)

static.mount(cheshire_cat_api)
# static files for hackable chat in cat/public
public.mount(cheshire_cat_api)


# error handling
@cheshire_cat_api.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"error": exc.errors()},
    )


# openapi customization
cheshire_cat_api.openapi = get_openapi_configuration_function(cheshire_cat_api)

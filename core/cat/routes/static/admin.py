import os
import re
import json
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse


def mount(cheshire_cat_api):

    mount_admin_index(cheshire_cat_api)

    # note html=False because index.html needs to be injected with runtime information
    cheshire_cat_api.mount(
        "/admin", StaticFiles(directory="/admin/", html=False), name="admin")


def mount_admin_index(cheshire_cat_api):

    @cheshire_cat_api.get("/admin/")
    def admin_index_injected():
        # admin index.html
        # all the admin files are served in a statci way
        # with the exception of index.html that needs a config file derived from core environments variable:
        #  - CORE_HOST
        #  - CORE_PORT
        #  - API_KEY
        cat_core_config = json.dumps({
            "CORE_HOST": os.getenv("CORE_HOST"),
            "CORE_PORT": os.getenv("CORE_PORT"),
            "CORE_USE_SECURE_PROTOCOLS": os.getenv("CORE_USE_SECURE_PROTOCOLS"),
            "API_KEY": os.getenv("API_KEY"),
        })

        # the admin sttic build is created during docker build from this repo:
        # https://github.com/cheshire-cat-ai/admin-vue
        # the files live inside the /admin folder (not visible in volume / cat code)
        with open("/admin/index.html", 'r') as f:
            html = f.read()

        # TODO: this is ugly, should be done with beautiful soup or a template
        regex = re.compile(
            r"catCoreConfig = (\{.*?\})", flags=re.MULTILINE | re.DOTALL)
        default_config = re.search(regex, html).group(1)
        html = html.replace(default_config, cat_core_config)


        return HTMLResponse(html)



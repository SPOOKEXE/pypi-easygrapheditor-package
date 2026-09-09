make a clone of comfyui node editor but as a gradio-style python package.

add integrations to gradio, streamlit, and other easy ui packages.

~/Documents/GitHub/node-graph-template

---

we split into two parts:
1. engine
2. ui

Engine does all the processing, asynchronous, defining value types, etc.
UI does all the adapter work and integrations into interfaces and whatnot - pygame, gradio, streamlit, etc.

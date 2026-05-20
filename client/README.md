# Hunyuan3D Client

Static browser client for the Hunyuan3D generation API.

## Use

Open `client/index.html` in a browser.

Default API address:

```text
http://159.48.242.5:56849
```

You can change the server IP, external port, and API key directly in the UI.
The values are saved in browser `localStorage`.

## Workflow

1. Click `Проверить сервер`.
2. Drag an image into the upload area or choose a file.
3. Click `Начать генерацию`.
4. Wait for `queued -> running -> done`.
5. Preview the returned GLB and download it.

The GLB preview uses Google's `model-viewer` web component from a CDN. If the
browser has no internet access, generation and download still work, but preview
may not render.

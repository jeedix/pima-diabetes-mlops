"""CLI: запуск FastAPI-сервиса (ЛР6)."""

import argparse

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Serve (ЛР6)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch", action="store_true", help="Использовать batch_service")
    args = parser.parse_args()

    app_str = "src.batch_service:app" if args.batch else "src.service:app"
    uvicorn.run(app_str, host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()

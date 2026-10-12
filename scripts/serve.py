import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="Start the MLO Inference REST API.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--reload", action="store_true", help="Enable hot reload.")
    args = parser.parse_args()

    print(f"Starting inference server on http://{args.host}:{args.port}...")
    uvicorn.run("mlo.pipeline.serving:app", host=args.host, port=args.port, reload=args.reload)

if __name__ == "__main__":
    main()

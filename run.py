import sys
import asyncio
import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # Use SelectorEventLoop on Windows to avoid signal handling issues
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            access_log=True,
            log_level="info"
        )
    except (KeyboardInterrupt, RuntimeError) as e:
        # Gracefully handle Windows signal interrupts
        print("\nServer shutting down...")
        sys.exit(0)

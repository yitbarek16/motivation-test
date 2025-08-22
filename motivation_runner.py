import json
import uuid
import logging
from datetime import datetime
from daily_motivation import (
    get_access_token, get_project_people,
    get_quote, enhance_quote, quote_overlay_on_image,
    upload_image_to_basecamp, post_message
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def main():
    # Load runner config (built in your workflow from runner_public.json + secrets)
    with open("runner_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    session_id = cfg["session_id"]
    account_id = int(cfg["account_id"])
    project_id = int(cfg["project_id"])
    message_board_id = int(cfg["message_board_id"])
    cc_ids = set(str(x) for x in cfg.get("cc_ids", []))  # string-compare for safety

    # Get access token (restored by your workflow step)
    access_token = get_access_token(session_id)
    if not access_token:
        logging.error("No access token available; exiting.")
        return

    # People
    all_people = get_project_people(account_id, project_id, access_token)
    main_people = [p for p in all_people if str(p["id"]) not in cc_ids]
    cc_people   = [p for p in all_people if str(p["id"]) in cc_ids]

    # Quote
    quote, author = get_quote()
    enhanced = enhance_quote(quote, author)

    # Fresh image every run (unique filename → avoids caching)
    base_image = "static/1.png"
    out_name = f"quote_{uuid.uuid4().hex}.png"
    out_path = f"static/{out_name}"
    quote_overlay_on_image(base_image, f"{quote} — {author}", output_path=out_path)

    # Upload to Basecamp → get SGID for inline embedding
    image_sgid = upload_image_to_basecamp(
        account_id=account_id,
        access_token=access_token,
        image_path=out_path
    )

    # Post (mentions at top with “Selam …”, CCs at bottom with “Cc: …”)
    ok = post_message(
        account_id=account_id,
        project_id=project_id,
        message_board_id=message_board_id,
        access_token=access_token,
        quote=quote,
        author=author,
        project_people=main_people,   # main mentions → top (“Selam …”)
        cc_people=cc_people,          # cc mentions → bottom (“Cc: …”)
        enhanced=enhanced,
        image_url=None,               # force SGID path
        image_sgid=image_sgid,        # <bc-attachment sgid="...">
        test_mode=False,
        mentions=None                 # let post_message build them
    )

    if ok:
        logging.info("Motivational message posted successfully.")
    else:
        logging.error("Failed to post motivational message.")

if __name__ == "__main__":
    main()

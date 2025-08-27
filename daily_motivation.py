import os
import re
import json
import time
import random
import html
import logging
import traceback
import textwrap
import requests
import schedule
import pytz
import urllib.parse
import webbrowser
from PIL import ImageFont

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import http.server
import socketserver


# Configuration
CLIENT_ID = "90ef5a48854d9eb4b98d49b7a6530247c7f180a2"
CLIENT_SECRET = "bba4f7c9ad0f4aab543f24cee9cb0897692166cb"

BASE_URL = "https://3.basecampapi.com"
TOKEN_FILE = lambda session_id: f"access_token_{session_id}.json"
REQUEST_TIMEOUT = 10
# Extend local token validity to minimize re-auth (adjust as needed)
TOKEN_EXPIRY = timedelta(days=3650)
USER_AGENT = "DailyMotivationApp/1.0"

# Initialize logging
try:
    logging.basicConfig(filename="motivational_poster.log", level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s", filemode="a")
except Exception as e:
    print(f"Failed to initialize logging: {e}")
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")


def retry_request(method_func, url, **kwargs):
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    try:
        response = session.request(method_func.__name__.split('.')[-1].upper(), url, **kwargs)
        return response
    except Exception as e:
        logging.error(f"Retry request failed for {url}: {e}")
        return None

def get_paginated_results(url: str, headers: Dict) -> List[Dict]:
    results = []
    current_url = url
    while current_url:
        try:
            response = retry_request(requests.get, current_url, headers=headers, timeout=REQUEST_TIMEOUT)
            if not response or response.status_code != 200:
                logging.error(f"Failed to fetch paginated results from {current_url}: HTTP {response.status_code if response else 'No response'}")
                return results
            data = response.json()
            if isinstance(data, list):
                results.extend(data)
            else:
                results.append(data)
            if 'Link' in response.headers:
                links = response.headers['Link'].split(',')
                next_link = next((link for link in links if 'rel="next"' in link), None)
                if next_link:
                    current_url = next_link.split(';')[0].strip('<>')
                else:
                    current_url = None
            else:
                current_url = None
        except Exception as e:
            logging.error(f"Error fetching paginated results from {current_url}: {e}")
            return results
    return results

def find_available_port(start_port: int = 8000, max_attempts: int = 100) -> int:
    port = start_port
    for _ in range(max_attempts):
        try:
            with socketserver.TCPServer(("localhost", port), None):
                return port
        except OSError:
            port += 1
    logging.error("No available ports found")
    raise RuntimeError("No available ports found")


def load_access_token(session_id: str) -> Optional[Dict]:
    try:
        with open(TOKEN_FILE(session_id), "r", encoding="utf-8") as f:
            data = json.load(f)
        expiry = datetime.fromisoformat(data["expiry"].replace('Z', '+00:00'))
        if datetime.now(timezone.utc) < expiry:
            return {"access_token": data["access_token"], "expiry": expiry}
        else:
            logging.info("Access token expired")
            os.remove(TOKEN_FILE(session_id))
            return None
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        logging.debug("Token file not found or invalid")
        return None


def get_account_info(access_token: str) -> Optional[int]:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": USER_AGENT}
    try:
        response = requests.get("https://launchpad.37signals.com/authorization.json",
                               headers=headers, timeout=REQUEST_TIMEOUT)
        if response.ok:
            data = response.json()
            accounts = data.get("accounts", [])
            if accounts:
                return accounts[0]["id"]
            logging.error("No accounts found in authorization response")
            return None
        logging.error(f"Failed to get account info: {response.text}")
        return None
    except Exception as e:
        logging.error(f"Error fetching account info: {e}")
        return None

def get_projects(account_id: int, access_token: str) -> List[Dict]:
    headers = {"Authorization": f"Bearer {access_token}", "User-Agent": USER_AGENT}
    url = f"{BASE_URL}/{account_id}/projects.json"
    try:
        response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.ok:
            projects = response.json()
            project_list = [{
                "name": project["name"],
                "id": project["id"],
                "message_board_id": next((dock["id"] for dock in project.get("dock", [])
                                         if dock["name"] == "message_board" and dock["enabled"]), None)
            } for project in projects]
            logging.info(f"Fetched {len(project_list)} projects for account_id {account_id}")
            return project_list
        logging.error(f"Failed to fetch projects: {response.text}")
        return []
    except Exception as e:
        logging.error(f"Error fetching projects: {e}")
        return []

def get_project_people(account_id: int, project_id: int, access_token: str) -> List[Dict]:
    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": USER_AGENT,
        "Accept": "application/json"
    }
    url = f"{BASE_URL}/{account_id}/projects/{project_id}/people.json"
    logging.debug(f"Fetching people from: {url} with account_id={account_id}, project_id={project_id}")
    try:
        response = retry_request(requests.get, url, headers=headers, timeout=REQUEST_TIMEOUT)
        if not response or response.status_code != 200:
            error_msg = f"Failed to fetch people for project {project_id}: HTTP {response.status_code if response else 'No response'}"
            logging.error(error_msg)
            return []

        people = get_paginated_results(url, headers)
        logging.info(f"Raw people API response for project {project_id}: {json.dumps(people, indent=2)}")
        valid_people = []
        for p in people:
            person_id = p.get("id")
            person_name = p.get("name")
            person_email = p.get("email_address", "N/A")
            sgid = p.get("attachable_sgid")
            logging.info(f"Processing person: ID={person_id}, Name={person_name}, Email={person_email}, SGID={sgid}, Data={p}")

            if not person_id:
                logging.warning(f"Skipping person with no ID: {p}")
                continue
            if not person_name:
                person_name = f"User_{person_id}"
                logging.warning(f"Person ID {person_id} has no name, using fallback: {person_name}")
            if not sgid:
                logging.warning(f"No attachable_sgid for person ID {person_id}, skipping for tagging")
                continue

            valid_people.append({
                "id": person_id,
                "name": person_name,
                "email_address": person_email,
                "sgid": sgid,
                "title": p.get("title", ""),
                "avatar_url": p.get("avatar_url", "https://bc3-production-assets-cdn.basecamp-static.com/default/avatar?v=1"),
                "company": p.get("company", {}).get("name", "N/A")
            })

        logging.info(f"Fetched {len(valid_people)} valid people for project {project_id}")
        logging.info(f"Valid people: {json.dumps(valid_people, indent=2)}")
        return valid_people
    except Exception as e:
        logging.error(f"Error fetching people for project {project_id}: {str(e)}\n{traceback.format_exc()}")
        return []

def format_mentions(person: dict) -> str:
    # Only the attachment tag. No figure/img/figcaption.
    sgid = person.get("sgid")
    if not sgid:
        return ""
    return (
        f'<bc-attachment sgid="{html.escape(sgid)}" '
        f'content-type="application/vnd.basecamp.mention"></bc-attachment>'
    )

def validate_image(image_path: str) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            img.load()
        file_size = os.path.getsize(image_path)
        if file_size > 5 * 1024 * 1024:  # 5MB limit
            logging.warning(f"Image too large: {file_size} bytes")
            return False
        if file_size == 0:
            logging.warning(f"Image file is empty: {image_path}")
            return False
        return True
    except Exception as e:
        logging.error(f"Invalid image file {image_path}: {e}")
        return False

def upload_image_to_basecamp(account_id: int, access_token: str, image_path: str) -> Optional[str]:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=3, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))

    if not validate_image(image_path):
        logging.error(f"Skipping upload due to invalid image: {image_path}")
        return None

    try:
        file_size = os.path.getsize(image_path)
        file_name = "quote_image.png"
        url = f"{BASE_URL}/{account_id}/attachments.json?name={urllib.parse.quote(file_name)}"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "User-Agent": USER_AGENT,
            "Content-Type": "image/png",
            "Content-Length": str(file_size)
        }
        logging.debug(f"Uploading image: {image_path}, size: {file_size} bytes, headers: {headers}")
        with open(image_path, "rb") as image_file:
            response = session.post(url, headers=headers, data=image_file.read(), timeout=REQUEST_TIMEOUT)
        if response.status_code == 201:
            data = response.json()
            attachable_sgid = data.get("attachable_sgid")
            if attachable_sgid:
                logging.info(f"Image uploaded to Basecamp, attachable_sgid: {attachable_sgid}")
                return attachable_sgid
            logging.error("No attachable_sgid in response")
            return None
        logging.error(f"Failed to upload image to Basecamp: {response.status_code} - {response.text[:500]}")
        return None
    except Exception as e:
        logging.error(f"Error uploading image to Basecamp: {e}")
        return None
    finally:
        if image_path.startswith("temp_quote_image_") and os.path.exists(image_path):
            try:
                os.remove(image_path)
                logging.debug(f"Cleaned up temporary image: {image_path}")
            except Exception as e:
                logging.error(f"Failed to clean up temporary image {image_path}: {e}")


def build_mentions(project_people: Optional[List[Dict]]) -> str:
    """
    Build mentions for a list of people, preserving input order.
    Returns formatted mentions string ready for HTML insertion.
    """
    logging.debug(f"build_mentions called with {len(project_people or [])} people")
    if not project_people:
        logging.debug("No project people, returning 'Team'")
        return "Team"

    seen_sgids = set()
    tags = []

    for p in project_people:  # 🚀 no sorting, preserve order
        sgid = p.get("sgid")
        if not sgid or sgid in seen_sgids:
            continue
        tag = format_mentions(p)
        if tag:
            tags.append(tag)
            seen_sgids.add(sgid)

    if tags:
        return " ".join(tags)
    else:
        return "Team"


def deduplicate_mentions_html(html: str) -> str:
    """
    Remove duplicate Basecamp mentions anywhere in the HTML, not just consecutive ones.
    Uniqueness is based on sgid.
    """
    seen_sgids = set()

    def repl(m):
        tag = m.group(0)
        sgid = m.group(1)
        if sgid in seen_sgids:
            return ""  # drop duplicate
        seen_sgids.add(sgid)
        return tag

    # Matches <bc-attachment ... sgid="..." ...> ... </bc-attachment>
    return re.sub(
        r'(<bc-attachment[^>]*\bsgid="([^"]+)"[^>]*>.*?</bc-attachment>)',
        lambda m: repl(m),
        html,
        flags=re.DOTALL
    )


def post_comment(
    account_id: int,
    project_id: int,
    parent_message_id: int,  # The dedicated thread’s message ID
    access_token: str,
    image_url: Optional[str] = None,
    image_sgid: Optional[str] = None,
    quote: Optional[str] = None,
    author: Optional[str] = None,
    project_people: Optional[List[Dict]] = None,
    cc_people: Optional[List[Dict]] = None,
    enhanced: Optional[str] = None
) -> bool:
    """
    Post a daily inspiration as a comment under an existing Basecamp message.
    """

    logging.debug("Attempting to post COMMENT to Basecamp")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "DailyMotivationApp/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8"
    }


    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/recordings/{parent_message_id}/comments.json"

    try:
        # --- Build main mentions ---
        main_mentions = ""
        if project_people:
            mention_html = build_mentions(project_people)
            if mention_html.strip():
                main_mentions = f"Good Morning {mention_html}"

        # --- Build CC mentions ---
        cc_mentions_html = ""
        if cc_people:
            cc_html = build_mentions(cc_people)
            if cc_html.strip():
                cc_mentions_html = f"<div><strong>Cc:</strong> <span>{cc_html}</span></div>"

        # --- Compose comment body ---
        content = ""
        if main_mentions:
            content += f"<p>{main_mentions}</p>"

        if image_sgid:
            content += f'<p><bc-attachment sgid="{image_sgid}"></bc-attachment></p>'
        elif image_url:
            content += f'<p><img src="{image_url}" alt="Motivational Quote" style="max-width:100%;"></p>'

        if enhanced:
            content += f"<p>{enhanced}</p>"

        # Footer
        content += '<br><div style="text-align:center; margin-top:10px;"><strong>Have a productive day!</strong></div>'

        if cc_mentions_html:
            content += cc_mentions_html

        # --- Deduplicate mentions ---
        content = deduplicate_mentions_html(content)

        # Debug log
        logging.debug("----- POST COMMENT CONTENT -----")
        logging.debug(content)
        logging.debug("----- END POST COMMENT -----")

        payload = {"content": content}

        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

        if response.ok:
            logging.info("Comment posted successfully")
            return True

        logging.error(f"Failed to post comment: {response.status_code} - {response.text[:300]}")
        return False

    except Exception as e:
        logging.error(f"Error posting comment: {str(e)}\n{traceback.format_exc()}")
        return False

import os
import json
import random
import logging
from openai import OpenAI

USED_QUOTES_FILE = "used_quotes.json"

# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

USED_QUOTES_FILE = "used_quotes.json"

def load_used_quotes() -> set:
    """Load previously used quotes (as (quote, author) tuples) from JSON file."""
    if not os.path.exists(USED_QUOTES_FILE):
        return set()
    try:
        with open(USED_QUOTES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            quotes = data.get("quotes", [])
            # convert list of dicts into tuples for easy lookup
            return set((q["quote"], q["author"]) for q in quotes if "quote" in q and "author" in q)
    except Exception:
        return set()

def save_used_quote(quote: str, author: str):
    """Save a new quote-author pair into used_quotes.json"""
    used = load_used_quotes()
    used.add((quote, author))

    # Convert back to list of dicts
    data = {"quotes": [{"quote": q, "author": a} for q, a in used]}
    with open(USED_QUOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def get_fallback_quote():
    fallback_quotes = [
        ("Discipline is the bridge between goals and accomplishment.", "Jim Rohn"),
        ("The secret of getting ahead is getting started.", "Mark Twain"),
        ("Push yourself, because no one else is going to do it for you.", "Anonymous"),
    ]
    return random.choice(fallback_quotes)


def get_quote():
    """
    Generates a motivational quote with an author using LLM.
    Ensures uniqueness with used_quotes.json.
    """
    used = load_used_quotes()

    prompt = """
Generate one (1) short, original, and unique motivational quote suitable for a professional workplace.
Tone & Style:
- Thoughtful, simple, and inspiring — like a gentle reminder that encourages reflection.
- Focus on discipline, growth, persistence, focus, teamwork, resilience, and purpose.
- Keep it short (max 20 words), deep, and easy to understand.
- It should sound meaningful enough that employees would reflect on it during their workday.

Content Rules:
- DO NOT include love, romance, or overly personal themes.
- DO NOT produce quotes that could encourage resignation, giving up, rebellion, laziness, or negativity.
- Quotes must encourage positive action, hope, and responsibility in a workplace context.
- Avoid clichés or generic internet-style phrases.

Format:
Quote text — Author/Annonymous
for example: Discipline is the bridge between goals and accomplishment. — Jim Rohn
Do not include any quotation marks, introductions, explanations, or closing remarks"""

    try:
        for _ in range(5):  # retry up to 5 times if duplicates
            completion = client.chat.completions.create(
                model="mistralai/mistral-small-3.2-24b-instruct:free",
                messages=[{"role": "user", "content": prompt}],
            )
            raw = completion.choices[0].message.content.strip()

            # Parse "Quote — Author"
            if "—" in raw:
                parts = raw.split("—", 1)
                quote = parts[0].strip().strip('"')
                author = parts[1].strip() if parts[1].strip() else "Anonymous"
            else:
                quote = raw.strip().strip('"')
                author = "Anonymous"

            # Ensure valid + unique
            if quote and quote not in used and not any(x in quote.lower() for x in ["love", "romance", "kiss", "heart"]):
                save_used_quote(quote, author)
                logging.info(f"Quote generated by LLM: '{quote}' — {author}")
                return quote, author

        logging.warning("Failed to generate unique quote, falling back")
        return get_fallback_quote()

    except Exception as e:
        logging.error(f"LLM quote generation failed: {e}")
        return get_fallback_quote()


# OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

def enhance_quote(quote, author):
    """
    Expands on the given quote with a short, uplifting message that inspires action and hope.
    """
    prompt = f"""Here's a motivational quote:
"{quote} — {author}"

Now expand on it with ,short motivational message for employees in the following format:
1. Start with a short, deep thought (1–2 sentences) that is simple, professional, and inspiring. 
   - Themes: discipline, growth, persistence, focus, teamwork, resilience, purpose, integrity. 
   - Avoid love, romance, politics, religion, or negativity.
2. Add 2–3 sentences expanding the thought in a reflective, workplace-appropriate tone. 
   - Should encourage responsibility, consistency, positivity, and self-improvement. 
   - Avoid clichés and overly casual expressions.
3. Provide 2–3 short, practical takeaways in bullet/emoji form (👉,💪, ✔︎). 
   - Each should be actionable, clear, and relevant to a professional environment.

Additional Safeguards:
- DO NOT generate anything that encourages resignation, rebellion, laziness, or passivity. 
- Keep the tone uplifting, professional, and aligned with workplace values. 
- Maximum length: 120 words.
- Do not repeat the original quote. Do not use quotation marks at the beginning or end.
- Feel free to add emojis (except heart emojis).
- Only return the only enhanced message. Do not include any introductions, explanations, or closing remarks.
- Do not include your intro like "Here's an enhanced message based on the quote provided:".
"""
    try:
        completion = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ERROR] Failed to enhance quote: {e}")
        return None

def quote_overlay_on_image(image_path, quote, output_path):
    """
    Overlays a quote and author on a given image and saves the result.
    Returns the path to the saved image.
    """
    try:
        # Load image
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        width, height = img.size

        # Use system font (always available on GitHub runners)
        font_path = "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"

        max_font_size = int(height * 0.08)
        min_font_size = int(height * 0.03)

        # Split quote and author
        if "—" in quote:
            quote_text, author_text = quote.rsplit("—", 1)
            quote_text = f"“{quote_text.strip()}”"   # quotation only around the quote
            author_text = f"— {author_text.strip()}"
        else:
            quote_text = quote
            quote_text = f"“{quote.strip()}”"
            author_text = ""

        # Fit quote text
        font_size = max_font_size
        while font_size >= min_font_size:
            quote_font = ImageFont.truetype(font_path, font_size)
            max_chars_per_line = int(width / (font_size * 0.7))
            wrapped_quote = textwrap.fill(quote_text.strip(), width=max_chars_per_line)
            quote_bbox = draw.multiline_textbbox((0, 0), wrapped_quote, font=quote_font, spacing=6)
            quote_width = quote_bbox[2] - quote_bbox[0]
            quote_height = quote_bbox[3] - quote_bbox[1]

            if quote_width <= width * 0.9 and quote_height <= height * 0.5:
                break
            font_size -= 2

        # Author font
        author_font_size = int(font_size * 0.6)
        author_font = ImageFont.truetype(font_path, author_font_size)
        author_bbox = draw.textbbox((0, 0), author_text, font=author_font)
        author_width = author_bbox[2] - author_bbox[0]
        author_height = author_bbox[3] - author_bbox[1]

        # Center positions
        total_height = quote_height + author_height + 20
        x_quote = (width - quote_width) / 2
        y_quote = (height - total_height) / 2
        x_author = (width - author_width) / 2
        y_author = y_quote + quote_height + 20

        # Draw text
        draw.multiline_text((x_quote, y_quote), wrapped_quote, font=quote_font, fill=(0, 0, 0), align="center", spacing=6)
        draw.text((x_author, y_author), author_text, font=author_font, fill=(30, 40, 70))

        # Resize for mobile
        max_width = 1080
        if width > max_width:
            scale = max_width / width
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.ANTIALIAS)

        # Save as PNG (overwrite)
        img = img.convert("RGB")
        img.save(output_path, format="PNG")
        return output_path

    except Exception as e:
        print(f"[ERROR] Failed to generate image: {e}")
        return None



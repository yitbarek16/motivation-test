import os
import re
import json
import time
import html
import logging
import traceback
import textwrap
import requests
import schedule
import pytz
import urllib.parse
import webbrowser

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, List
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import http.server
import socketserver


load_dotenv() # Load environment variables

# Configuration
CLIENT_ID = "90ef5a48854d9eb4b98d49b7a6530247c7f180a2"
CLIENT_SECRET = "bba4f7c9ad0f4aab543f24cee9cb0897692166cb"
REDIRECT_URI = "http://localhost:5000/oauth/callback"
BASE_URL = "https://3.basecampapi.com"
TOKEN_FILE = lambda session_id: f"access_token_{session_id}.json"
REQUEST_TIMEOUT = 10
# Extend local token validity to minimize re-auth (adjust as needed)
TOKEN_EXPIRY = timedelta(days=3650)
#PEXELS_API_KEY = "Zum5sloqdAGsnMHFm4ICOmEDAxZ4O2tujTCResgQWMug7iGQ7b2DFkbh"
USER_AGENT = "DailyMotivationApp/1.0"
#QUOTABLE_API_URL = "https://api.quotable.io/random?tags=leadership|success|wisdom||development|resilience|intelligence"
#ZENQUOTES_API_URL = "https://zenquotes.io/api/quotes"
EAT_TZ = pytz.timezone("Africa/Nairobi")  # EAT is UTC+3


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

def save_access_token(access_token: str, expiry: datetime, session_id: str):
    try:
        data = {"access_token": access_token, "expiry": expiry.isoformat()}
        with open(TOKEN_FILE(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f)
        logging.info("Access token saved successfully")
    except Exception as e:
        logging.error(f"Failed to save access token: {e}")

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

def get_access_token(session_id: str) -> Optional[str]:
    token_data = load_access_token(session_id)
    if token_data and token_data.get("access_token"):
        logging.debug("Using existing access token")
        return token_data["access_token"]
    port = find_available_port()
    AUTH_URL = f"https://launchpad.37signals.com/authorization/new?type=web_server&client_id={CLIENT_ID}&redirect_uri={REDIRECT_URI}"
    class OAuthHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            try:
                if self.path.startswith('/oauth/callback'):
                    params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                    code = params.get('code', [None])[0]
                    if code:
                        token_response = requests.post(
                            "https://launchpad.37signals.com/authorization/token.json",
                            data={
                                "type": "web_server",
                                "client_id": CLIENT_ID,
                                "client_secret": CLIENT_SECRET,
                                "redirect_uri": REDIRECT_URI,
                                "code": code
                            },
                            timeout=REQUEST_TIMEOUT
                        )
                        if token_response.ok:
                            token_data = token_response.json()
                            access_token = token_data.get("access_token")
                            if access_token:
                                expiry = datetime.now(timezone.utc) + TOKEN_EXPIRY
                                save_access_token(access_token, expiry, session_id)
                                self.respond_with("Success! You can close this tab.")
                            else:
                                error_msg = token_data.get("error", "No access token")
                                logging.error(f"Token exchange failed: {error_msg}")
                                self.respond_with(f"Token exchange failed: {error_msg}")
                        else:
                            error_msg = token_response.text
                            logging.error(f"Token exchange failed: {error_msg}")
                            self.respond_with(f"Token exchange failed: {error_msg}")
                    else:
                        error_msg = params.get('error', ['No code received'])[0]
                        logging.error(f"OAuth callback error: {error_msg}")
                        self.respond_with(f"Authentication failed: {error_msg}")
                else:
                    logging.error(f"Invalid callback URL: {self.path}")
                    self.respond_with("Invalid callback URL")
            except Exception as e:
                logging.error(f"OAuth handler error: {e}")
                self.respond_with(f"Authentication error: {e}")
        def respond_with(self, message):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(f"<html><body><h1>{message}</h1></body></html>".encode())
    try:
        logging.info("Initiating OAuth flow")
        webbrowser.open(AUTH_URL)
        with socketserver.TCPServer(("localhost", port), OAuthHandler) as httpd:
            httpd.timeout = 120
            httpd.handle_request()
    except Exception as e:
        logging.error(f"OAuth flow failed: {e}")
        return None
    token_data = load_access_token(session_id)
    return token_data.get("access_token") if token_data else None

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
    if not project_people:
        return "Selam Team,"

    # Stable order by casefolded name; dedupe strictly by sgid
    seen_sgids = set()
    tags = []

    for p in sorted(project_people, key=lambda x: x.get("name", "").casefold()):
        sgid = p.get("sgid")
        if not sgid or sgid in seen_sgids:
            continue
        tag = format_mentions(p)
        if tag:
            tags.append(tag)
            seen_sgids.add(sgid)

    return f"{' '.join(tags)}," if tags else "Selam Team,"

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

def post_message(
    account_id: int,
    project_id: int,
    message_board_id: int,
    access_token: str,
    image_url: Optional[str] = None,
    image_sgid: Optional[str] = None,
    mentions: Optional[str] = None,
    quote: Optional[str] = None,
    author: Optional[str] = None,
    test_mode: bool = False,
    project_people: Optional[List[Dict]] = None,   # Main mention list
    cc_people: Optional[List[Dict]] = None,        # CC recipients
    enhanced: Optional[str] = None
) -> bool:
    """
    Post a message to a Basecamp message board with deduplicated mentions.
    Debug-friendly: logs everything.
    """
    logging.debug("Attempting to post message to Basecamp")

    headers = {
        "Authorization": f"Bearer {access_token}",
        "User-Agent": "DailyMotivationApp/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json; charset=utf-8"
    }

    url = f"{BASE_URL}/{account_id}/buckets/{project_id}/message_boards/{message_board_id}/messages.json"

    try:
        # Build main mentions if not provided
        if not mentions and project_people:
            mentions = build_mentions(project_people)
        elif not mentions:
            mentions = ""

        logging.debug(f"Main mentions: {mentions}")
        logging.debug(f"CC people: {[p['name'] for p in (cc_people or [])]}")

        # Start HTML content
        content = ""
        if mentions.strip():
            content += f"<p>Selam {mentions}</p>"

        # Embed image
        if image_sgid:
            content += f'<p><bc-attachment sgid="{image_sgid}"></bc-attachment></p>'
        elif image_url:
            content += f'<p><img src="{image_url}" alt="Motivational Quote" style="max-width:100%;"></p>'

        # Enhanced message
        if enhanced:
            content += f"<p>{enhanced}</p>"

        # Footer
        content += '<br><div style="text-align:center; margin-top:10px;"><strong>Have a productive day!</strong></div>'

        # Add CCs below footer (exclude Selam)
        if cc_people:
            cc_mentions_html = build_mentions(cc_people)
            if cc_mentions_html.strip():
                content += f'<div style="margin-top:10px;"><strong>Cc:</strong> {cc_mentions_html}</div>'

        # Deduplicate mentions
        content = deduplicate_mentions_html(content)

        logging.debug("----- POST CONTENT TO BASECAMP -----")
        logging.debug(content)
        logging.debug("----- END POST CONTENT -----")

        payload = {
            "subject": "Daily Inspiration",
            "content": content,
            "status": "active"
        }

        response = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

        if response.ok:
            logging.info("Message posted successfully")
            return True

        logging.error(f"Failed to post message: {response.status_code}\n{response.text}")
        return False

    except Exception as e:
        logging.error(f"Error posting message: {str(e)}\n{traceback.format_exc()}")
        return False

def schedule_daily_post(
    account_id: int,
    project_id: int,
    message_board_id: int,
    access_token: str,
    schedule_time: str,
    cc_people: Optional[List[Dict]] = None,
    test_mode: bool = False,
    project_people: Optional[List[Dict]] = None
):
    schedule.clear()

    def job():
        now = datetime.now(EAT_TZ)
        if not test_mode and now.weekday() >= 5:
            logging.info(f"Skipping post on {now.strftime('%A')} (weekend)")
            return

        logging.info(f"Running scheduled post at {now.strftime('%Y-%m-%d %H:%M:%S')}")

        # Main people = project_people - cc_people
        cc_ids = {str(p["id"]) for p in (cc_people or [])}
        if project_people:
            main_people = [p for p in project_people if str(p["id"]) not in cc_ids]
        else:
            main_people = get_project_people(account_id, project_id, access_token)
            main_people = [p for p in main_people if str(p["id"]) not in cc_ids]

        # Clean CCs (exclude "Selam")
        filtered_cc_people = [
            p for p in (cc_people or [])
            if p.get("name", "").strip().lower() != "selam"
        ]

        # Get quote and enhanced version
        quote, author = get_quote()
        enhanced = enhance_quote(quote, author)

        # Generate image locally
        base_image = "static/1.png"
        output_filename = "img1.png"
        output_image = f"static/{output_filename}"
        quote_overlay_on_image(base_image, f"{quote} — {author}", output_path=output_image)

        # Upload image to Basecamp
        from daily_motivation import upload_image_to_basecamp
        attachable_sgid = upload_image_to_basecamp(
            account_id=account_id,
            access_token=access_token,
            image_path=output_image
        )

        # ✅ Use the same format as test_post (post_message handles layout)
        success = post_message(
            account_id=account_id,
            project_id=project_id,
            message_board_id=message_board_id,
            access_token=access_token,
            quote=quote,
            author=author,
            test_mode=test_mode,
            project_people=main_people,       # mentions above
            cc_people=filtered_cc_people,     # CCs below footer
            enhanced=enhanced,
            image_url=None,
            image_sgid=attachable_sgid
        )

        if success:
            logging.info("Scheduled message posted successfully")
        else:
            logging.error("Scheduled message failed to post")

    # ✅ Test mode: just run once immediately
    if test_mode:
        job()
        return

    # Real daily scheduling
    try:
        datetime.strptime(schedule_time, "%H:%M")
        schedule.every().day.at(schedule_time).do(job)
        logging.info(f"Scheduled daily post at {schedule_time} EAT (Monday–Friday)")
    except ValueError:
        logging.error(f"Invalid time format for scheduling: {schedule_time}")
        return

    # Run scheduler loop
    while True:
        schedule.run_pending()
        time.sleep(60)



def get_quote():
    """
    Generates a motivational quote with an author using the Mistral LLM.
    Falls back to 'Anonymous' if no clear author is provided.
    """
    themes = [
    "Accountability", "Strategic Focus", "Continuous Growth", "Adaptability",       
    "Curiosity & Innovation", "Discipline", "Growth", "Resilience", "Teamwork",
            ]
    prompt = f"""
    Give me one short motivational quote about {themes}.
    It should be original, creatively phrased, and not a famous or widely circulated quote.
    Avoid quotes by Steve Jobs, Einstein, or other well-known figures.
    Include the author's name after the quote, separated by a dash. If unknown, use 'Anonymous' separated by a dash.
    Do not include introductions, or explanations.

    Today's date is {datetime.now(EAT_TZ).strftime('%A, %B %d')}. Make the quote feel fresh and relevant to this day.
    change the quote every day to keep it unique and engaging.
    Example format: "The only way to do great work is to love what you do" — Steve Jobs
    """

    try:
        completion = client.chat.completions.create(
            model="mistralai/mistral-small-3.2-24b-instruct:free",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        response = completion.choices[0].message.content.strip()

        # Split quote and author
        if "—" in response:
            quote, author = map(str.strip, response.split("—", 1))
            # Fallback if author is missing or generic
            if not author or author.lower() in ["unknown", "anonymous"]:
                author = "Anonymous"
        else:
            quote = response
            author = "Anonymous"

        return quote, author

    except Exception as e:
        print(f"[ERROR] Failed to generate quote: {e}")
        return None, None

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

Now expand on it with a short, uplifting message that inspires action and hope. 
Do not repeat the original quote. Do not use quotation marks at the beginning or end.
Feel free to add emojis (except heart emojis).
Only return the enhanced message. Do not include any introductions, explanations, or closing remarks.
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

        # Font settings
        max_font_size = int(height * 0.08)
        min_font_size = int(height * 0.03)
        font_path = "times.ttf"  # Adjust path if needed

        # Split quote and author
        if "—" in quote:
            quote_text, author_text = quote.rsplit("—", 1)
            author_text = f"— {author_text.strip()}"
        else:
            quote_text = quote
            author_text = ""

        # Fit quote text
        font_size = max_font_size
        while font_size >= min_font_size:
            quote_font = ImageFont.truetype(font_path, font_size)
            max_chars_per_line = int(width / (font_size * 0.7))
            quoted_text = f'{quote_text.strip()}'
            wrapped_quote = textwrap.fill(quoted_text, width=max_chars_per_line)
            quote_bbox = draw.multiline_textbbox((0, 0), wrapped_quote, font=quote_font, spacing=6)
            quote_width = quote_bbox[2] - quote_bbox[0]
            quote_height = quote_bbox[3] - quote_bbox[1]

            if quote_width <= width * 0.9 and quote_height <= height * 0.5:
                break
            font_size -= 2

        # Author font (smaller)
        author_font_size = int(font_size * 0.6)
        author_font = ImageFont.truetype(font_path, author_font_size)
        author_bbox = draw.textbbox((0, 0), author_text, font=author_font)
        author_width = author_bbox[2] - author_bbox[0]
        author_height = author_bbox[3] - author_bbox[1]

        # Total height
        total_height = quote_height + author_height + 20

        # Center position
        x_quote = (width - quote_width) / 2
        y_quote = (height - total_height) / 2
        x_author = (width - author_width) / 2
        y_author = y_quote + quote_height + 20

        # Colors
        color1 = (25, 25, 112)  # quote
        color2 = (30, 40, 70)   # author

        # Draw text
        draw.multiline_text((x_quote, y_quote), wrapped_quote, font=quote_font, fill=color1, align="center", spacing=6)
        draw.text((x_author, y_author), author_text, font=author_font, fill=color2)

        # Resize for mobile (max width 1080px)
        max_width = 1080
        if width > max_width:
            scale = max_width / width
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, Image.ANTIALIAS)

        # Convert to RGB to avoid transparency issues
        img = img.convert("RGB")

        # Save as PNG
        img.save(output_path, format="PNG")
        return output_path

    except Exception as e:
        print(f"[ERROR] Failed to generate image: {e}")
        return None


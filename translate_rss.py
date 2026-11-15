import html
import json
import os
import re
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from time import mktime

import feedparser
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from openai import OpenAI

DATA_FILE = "data/articles.json"
FEED_FILE = "docs/feed.xml"
PICKER_FILE = "docs/sec.html"
MAX_ARTICLES = 60
MAX_INPUT_CHARS = 28000

SYSTEM_PROMPT = """Sen nörobilim, bilişsel bilimler ve psikoloji alanlarında uzmanlaşmış, akademik Türkçeye tam hakim profesyonel bir bilim çevirmenisin.

Görevin, İngilizce bir e-posta bültenini Türkçeye doğru, akıcı ve terminolojiye sadık kalarak çevirmektir.

Kurallar:
1. Alan Terminolojisi: Kavramları günlük dildeki karşılıklarıyla değil, Türkçedeki yerleşik akademik karşılıklarıyla çevir.
   - Örnekler: "affect" -> duygulanım, "executive function" -> yürütücü işlev, "conditioning" -> koşullanma, "reinforcement" -> pekiştirme, "cognition" -> biliş, "working memory" -> çalışma belleği, "neuroplasticity" -> nöroplastisite, "salience" -> belirginlik, "default mode network" -> varsayılan mod ağı.
2. Orijinal Terim Desteği: Önemli bilimsel terimlerin, beyin bölgelerinin veya kuramların metinde ilk geçtiği yerde parantez içinde İngilizcesini bırak. Örn: *yürütücü işlevler (executive functions)*.
3. Linkleri Koru: Metin içinde `[metin](url)` formatında verilen bilimsel makale, araştırma ve kaynak bağlantılarını mutlaka `<a href="url">çevrilmiş metin</a>` şeklinde HTML linki olarak koru.
4. Gereksiz Kısımları Temizle: Bülten sonundaki "unsubscribe / abonelikten çık", sosyal medya ikon yazıları, reklam/sponsorluk uyarıları gibi bülten dışı kalıpları çevirme, atla.
5. Çıktı Formatı:
   - İlk satıra SADECE şu formatta çevrilmiş başlığı yaz:
     BASLIK: <Bültenin Türkçe Başlığı>
   - Bir boş satır bırak, ardından doğrudan temiz HTML gövdesini başlat.
   - Gövdenin en başına `<h3>📌 Bu Bültende Öne Çıkanlar</h3>` başlığı altında 3 maddelik (`<ul><li>...</li></ul>`) net bir özet ekle, ardından `<hr>` koyup bültenin tam ve eksiksiz çevirisini ver.
   - Yalnızca şu basit HTML etiketlerini kullan: `<h3>`, `<h4>`, `<p>`, `<ul>`, `<ol>`, `<li>`, `<strong>`, `<em>`, `<blockquote>`, `<a href="...">`, `<hr>`.
   - Yanıtını asla ```html gibi kod blokları içine alma, doğrudan düz metin olarak ver."""


def clean_html_for_llm(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")

    for tag in soup(["script", "style", "head", "meta", "noscript", "svg"]):
        tag.decompose()

    for img in soup.find_all("img"):
        w = img.get("width", "")
        h = img.get("height", "")
        if w in ("0", "1") or h in ("0", "1"):
            img.decompose()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if "kill-the-newsletter.com/feeds/" in href:
            a.decompose()
            continue
        text = a.get_text(" ", strip=True)
        if text and href.startswith(("http://", "https://")):
            a.replace_with(f"[{text}]({href})")
        elif text:
            a.replace_with(text)
        else:
            a.decompose()

    text = soup.get_text(separator="\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)

    if len(cleaned) > MAX_INPUT_CHARS:
        cleaned = cleaned[:MAX_INPUT_CHARS] + "\n\n[Not: Metin uzunluk sınırı nedeniyle burada sonlandırıldı.]"
    return cleaned


def strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:html)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_llm_output(output: str, fallback_title: str) -> tuple[str, str]:
    output = strip_code_fences(output)
    lines = output.splitlines()
    title = fallback_title
    body_start_idx = 0

    for i, line in enumerate(lines[:5]):
        if line.strip().upper().startswith("BASLIK:"):
            parsed_title = line.split(":", 1)[1].strip()
            if parsed_title:
                title = parsed_title
            body_start_idx = i + 1
            break

    body = "\n".join(lines[body_start_idx:]).strip()
    body = re.sub(r"^ICERIK:\s*", "", body, flags=re.IGNORECASE).strip()
    return title, body


def translate_with_gemini(title: str, content: str, system_prompt: str = SYSTEM_PROMPT) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None

    client = genai.Client(api_key=api_key)
    user_prompt = f"Orijinal Başlık: {title}\n\nİçerik:\n{content}"

    attempts = [
        (
            "gemini-3.8-flash",
            types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                thinking_config=types.ThinkingConfig(
                    thinking_level=types.ThinkingLevel.LOW
                ),
            ),
        ),
        (
            "gemini-3.5-flash-lite",
            types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
        ),
    ]

    for model_name, config in attempts:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=config,
            )
            if response and response.text:
                print(f"Çeviri başarılı ({model_name}): {title}")
                return response.text
        except Exception as e:
            print(f"Gemini ({model_name}) hatası: {e}")

    return None


def translate_with_groq(title: str, content: str, system_prompt: str = SYSTEM_PROMPT) -> str | None:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None

    try:
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        user_prompt = f"Orijinal Başlık: {title}\n\nİçerik:\n{content[:20000]}"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content
        if text:
            print(f"Yedek çeviri başarılı (Groq Llama 3.3 70B): {title}")
            return text
    except Exception as e:
        print(f"Groq hatası: {e}")

    return None


def translate_article(title: str, raw_html: str) -> tuple[str, str] | None:
    cleaned_text = clean_html_for_llm(raw_html)
    if not cleaned_text:
        return None

    raw_output = translate_with_gemini(title, cleaned_text)
    if not raw_output:
        raw_output = translate_with_groq(title, cleaned_text)

    if not raw_output:
        return None

    return parse_llm_output(raw_output, title)


def load_articles() -> list[dict]:
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_articles(articles: list[dict]) -> None:
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(articles[:MAX_ARTICLES], f, ensure_ascii=False, indent=2)


def cdata_safe(text: str) -> str:
    return text.replace("]]>", "]]]]><![CDATA[>")


def build_rss(articles: list[dict]) -> None:
    os.makedirs("docs", exist_ok=True)

    with open("docs/robots.txt", "w", encoding="utf-8") as f:
        f.write("User-agent: *\nDisallow: /\n")

    now_rfc = format_datetime(datetime.now(timezone.utc))
    items_xml = []

    for art in articles[:MAX_ARTICLES]:
        dt = datetime.fromisoformat(art["published_iso"])
        pub_rfc = format_datetime(dt)
        title_esc = html.escape(art["title"])
        link_esc = html.escape(art["link"])
        guid_esc = html.escape(art["id"])
        body_html = cdata_safe(art["content_html"])

        items_xml.append(
            f"""    <item>
      <title>{title_esc}</title>
      <link>{link_esc}</link>
      <guid isPermaLink="false">{guid_esc}</guid>
      <pubDate>{pub_rfc}</pubDate>
      <description><![CDATA[{body_html}]]></description>
    </item>"""
        )

    rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Nörobilim &amp; Psikoloji Bültenleri (TR)</title>
    <link>https://kill-the-newsletter.com/</link>
    <description>Akademik terminolojiye sadık otomatik Türkçe bülten akışı</description>
    <language>tr</language>
    <lastBuildDate>{now_rfc}</lastBuildDate>
{chr(10).join(items_xml)}
  </channel>
</rss>
"""
    with open(FEED_FILE, "w", encoding="utf-8") as f:
        f.write(rss_xml)


def is_ignored_link(href: str, text: str) -> bool:
    low_href = href.lower()
    low_text = text.lower()
    skip_domains = (
        "kill-the-newsletter.com",
        "mail-settings.google.com",
        "support.google.com",
        "unsubscribe",
        "mailto:",
    )
    if any(k in low_href for k in skip_domains):
        return True
    if any(w in low_text for w in ("abonelikten", "unsubscribe", "iptal et")):
        return True
    return len(text.strip()) < 4


def build_picker_html(articles: list[dict]) -> None:
    os.makedirs("docs", exist_ok=True)
    repo_name = os.environ.get("GITHUB_REPOSITORY", "")
    tr_tz = timezone(timedelta(hours=3))

    already_translated = set()
    for art in articles:
        if art["id"].startswith("article:"):
            already_translated.add(art["link"])
            if art.get("source_url"):
                already_translated.add(art["source_url"])

    sections = []
    nl_count = 0

    for art in articles:
        if art["id"].startswith("article:"):
            continue
        if nl_count >= 10:
            break

        soup = BeautifulSoup(art["content_html"], "html.parser")
        seen_hrefs = set()
        link_rows = []

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(" ", strip=True)
            if not href.startswith(("http://", "https://")):
                continue
            if href in seen_hrefs or is_ignored_link(href, text):
                continue
            seen_hrefs.add(href)

            done = href in already_translated
            badge = ' <span class="badge">✓ Çevrildi</span>' if done else ""
            disabled = " disabled" if done else ""
            href_esc = html.escape(href)
            text_esc = html.escape(text)
            domain = re.sub(r"^https?://(www\.)?", "", href).split("/")[0]

            link_rows.append(
                f"""<label class="row{' done' if done else ''}">
  <input type="checkbox" name="url" value="{href_esc}"{disabled}>
  <span>
    <strong>{text_esc}</strong>{badge}
    <small>{html.escape(domain)}</small>
  </span>
</label>"""
            )

        if link_rows:
            nl_count += 1
            dt = datetime.fromisoformat(art["published_iso"]).astimezone(tr_tz)
            date_str = dt.strftime("%d.%m.%Y")
            nl_title = html.escape(art["title"])
            sections.append(
                f"""<section class="group">
  <h2>{nl_title} <span class="dt">{date_str}</span></h2>
  {chr(10).join(link_rows)}
</section>"""
            )

    if not sections:
        sections.append('<p class="empty">Henüz bültenlerden çıkarılmış makale linki yok.</p>')

    picker_html = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex,nofollow">
  <title>Tam Makale Çevirisi Seç</title>
  <style>
    :root {{ --bg:#f8fafc; --card:#fff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0; --btn:#16a34a; }}
    @media (prefers-color-scheme: dark) {{
      :root {{ --bg:#0f172a; --card:#1e293b; --text:#e2e8f0; --muted:#94a3b8; --border:#334155; --btn:#22c55e; }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; padding:20px 14px 96px; background:var(--bg); color:var(--text); font-family:system-ui,-apple-system,sans-serif; }}
    main {{ max-width:700px; margin:0 auto; }}
    h1 {{ font-size:1.3rem; margin:0 0 6px; }}
    p.sub {{ color:var(--muted); font-size:0.9rem; margin:0 0 20px; }}
    .group {{ background:var(--card); border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:16px; }}
    .group h2 {{ font-size:1rem; margin:0 0 12px; border-bottom:1px solid var(--border); padding-bottom:8px; }}
    .dt {{ font-size:0.8rem; color:var(--muted); font-weight:normal; margin-left:6px; }}
    label.row {{ display:flex; align-items:flex-start; gap:10px; padding:8px 6px; border-radius:6px; cursor:pointer; }}
    label.row:hover {{ background:rgba(100,116,139,0.08); }}
    label.row.done {{ opacity:0.55; cursor:default; }}
    input[type="checkbox"] {{ width:18px; height:18px; margin-top:3px; flex-shrink:0; }}
    label.row span {{ display:flex; flex-direction:column; }}
    label.row small {{ color:var(--muted); font-size:0.78rem; margin-top:2px; }}
    .badge {{ display:inline-block; font-size:0.72rem; background:#dcfce7; color:#166534; padding:1px 6px; border-radius:4px; margin-left:6px; }}
    textarea {{ width:100%; padding:10px; border:1px solid var(--border); border-radius:8px; background:var(--card); color:var(--text); font-family:inherit; }}
    .bar {{ position:fixed; bottom:0; left:0; right:0; background:var(--card); border-top:1px solid var(--border); padding:12px 16px; display:flex; justify-content:center; }}
    button {{ background:var(--btn); color:#fff; border:0; border-radius:8px; padding:12px 24px; font-size:1rem; font-weight:600; cursor:pointer; width:100%; max-width:700px; }}
  </style>
</head>
<body>
  <main>
    <h1>📚 Tam Makale Seçimi</h1>
    <p class="sub">Tam metnini Feeder'a eklemek istediğiniz makaleleri işaretleyin ve alttaki butona basın.</p>
    {chr(10).join(sections)}
    <div class="group">
      <h2>Ekstra Link (İsteğe Bağlı)</h2>
      <textarea id="extra-urls" rows="2" placeholder="Bülten dışından çevirmek istediğiniz bir link varsa buraya yapıştırabilirsiniz..."></textarea>
    </div>
  </main>
  <div class="bar">
    <button onclick="submitSelected()">Seçilenleri Çevir (GitHub'a Gönder)</button>
  </div>
  <script>
    function submitSelected() {{
      const checked = Array.from(document.querySelectorAll('input[name="url"]:checked')).map(el => el.value);
      const extra = document.getElementById('extra-urls').value
        .split(/\\s+/)
        .map(s => s.trim())
        .filter(s => s.startsWith('http://') || s.startsWith('https://'));
      const allUrls = [...new Set([...checked, ...extra])];
      if (allUrls.length === 0) {{
        alert('Lütfen en az bir makale seçin veya link yapıştırın.');
        return;
      }}
      const title = encodeURIComponent('Çevir: ' + allUrls.length + ' Makale');
      const body = encodeURIComponent(allUrls.join('\\n'));
      const repo = "{html.escape(repo_name)}";
      window.location.href = `https://github.com/${{repo}}/issues/new?title=${{title}}&body=${{body}}`;
    }}
  </script>
</body>
</html>
"""
    with open(PICKER_FILE, "w", encoding="utf-8") as f:
        f.write(picker_html)


def main() -> None:
    feed_urls_raw = os.environ.get("KTN_FEED_URL", "").strip()
    if not feed_urls_raw:
        raise RuntimeError("KTN_FEED_URL secret tanımlı değil!")

    feed_urls = [u.strip() for u in feed_urls_raw.split(",") if u.strip()]
    existing_articles = load_articles()
    seen_ids = {a["id"] for a in existing_articles}
    new_articles = []

    for url in feed_urls:
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            entry_id = entry.get("id") or entry.get("link") or entry.get("title")
            if not entry_id or entry_id in seen_ids:
                continue

            orig_title = entry.get("title", "Başlıksız Bülten")
            raw_html = ""
            if entry.get("content"):
                raw_html = entry["content"][0].get("value", "")
            elif entry.get("summary"):
                raw_html = entry["summary"]

            if not raw_html:
                continue

            translated = translate_article(orig_title, raw_html)
            if not translated:
                print(f"Çevrilemedi, sonraki çalışmada tekrar denenecek: {orig_title}")
                continue

            tr_title, tr_html = translated
            time_struct = entry.get("published_parsed") or entry.get("updated_parsed")
            if time_struct:
                dt = datetime.fromtimestamp(mktime(time_struct), tz=timezone.utc)
            else:
                dt = datetime.now(timezone.utc)

            new_articles.append(
                {
                    "id": entry_id,
                    "title": tr_title,
                    "original_title": orig_title,
                    "link": entry.get("link", "https://kill-the-newsletter.com/"),
                    "published_iso": dt.isoformat(),
                    "content_html": tr_html,
                }
            )
            seen_ids.add(entry_id)

    if new_articles:
        combined = new_articles + existing_articles
        combined.sort(key=lambda x: x["published_iso"], reverse=True)
        save_articles(combined)
        build_rss(combined)
        build_picker_html(combined)
        print(f"{len(new_articles)} yeni bülten çevrildi ve feed güncellendi.")
    else:
        if not os.path.exists(FEED_FILE):
            build_rss(existing_articles)
        build_picker_html(existing_articles)
        print("Yeni bülten bulunamadı.")


if __name__ == "__main__":
    main()
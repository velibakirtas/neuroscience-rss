import os
import re
from datetime import datetime, timezone

import requests
import trafilatura
from bs4 import BeautifulSoup

from translate_rss import (
    build_picker_html,
    build_rss,
    load_articles,
    parse_llm_output,
    save_articles,
    translate_with_gemini,
    translate_with_groq,
)

MAX_ARTICLE_CHARS = 36000

ARTICLE_SYSTEM_PROMPT = """Sen nörobilim, bilişsel bilimler ve psikoloji alanlarında uzmanlaşmış, akademik Türkçeye tam hakim profesyonel bir bilim çevirmenisin.

Görevin, sana verilen tam metinli bilimsel makale veya yazıyı anlam kaybı olmadan, akıcı ve terminolojiye sadık kalarak Türkçeye çevirmektir.

Kurallar:
1. Alan Terminolojisi: Kavramları Türkçedeki yerleşik akademik karşılıklarıyla çevir (ör. "affect" -> duygulanım, "executive function" -> yürütücü işlev, "conditioning" -> koşullanma, "cognition" -> biliş, "working memory" -> çalışma belleği, "neuroplasticity" -> nöroplastisite).
2. Orijinal Terim Desteği: Önemli bilimsel terimlerin, beyin bölgelerinin veya kuramların metinde ilk geçtiği yerde parantez içinde İngilizcesini bırak.
3. Linkleri Koru: Metindeki bilimsel atıf ve kaynak bağlantılarını `<a href="url">metin</a>` şeklinde koru.
4. Çıktı Formatı:
   - İlk satıra SADECE şu formatta çevrilmiş başlığı yaz:
     BASLIK: <Makalenin Türkçe Başlığı>
   - Bir boş satır bırak, ardından doğrudan temiz HTML gövdesini başlat.
   - En başa `<h3>📌 Makaleden Öne Çıkanlar</h3>` altında 3 maddelik (`<ul><li>...</li></ul>`) özet koy, `<hr>` çek ve makalenin tam çevirisini alt başlıklarıyla birlikte ver.
   - Yalnızca şu HTML etiketlerini kullan: `<h3>`, `<h4>`, `<p>`, `<ul>`, `<ol>`, `<li>`, `<strong>`, `<em>`, `<blockquote>`, `<a href="...">`, `<hr>`.
   - Yanıtını asla ```html kod blokları içine alma."""


def fetch_article(url: str) -> tuple[str, str, str] | None:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    resp = requests.get(url, headers=headers, timeout=25, allow_redirects=True)
    resp.raise_for_status()
    final_url = resp.url
    raw_html = resp.text

    extracted = trafilatura.extract(
        raw_html,
        url=final_url,
        output_format="markdown",
        include_links=True,
        include_formatting=True,
        favor_recall=True,
    )
    meta = trafilatura.extract_metadata(raw_html, default_url=final_url)
    title = meta.title if meta and meta.title else ""

    if not title:
        soup = BeautifulSoup(raw_html, "html.parser")
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
    if not title:
        title = final_url

    if not extracted or len(extracted.strip()) < 250:
        return None

    return final_url, title, extracted[:MAX_ARTICLE_CHARS]


def main() -> None:
    raw_input = "\n".join(
        [
            os.environ.get("INPUT_URLS", ""),
            os.environ.get("ISSUE_BODY", ""),
        ]
    )
    urls = list(dict.fromkeys(re.findall(r"https?://[^\s\)\]>\"']+", raw_input)))

    if not urls:
        with open("result_summary.txt", "w", encoding="utf-8") as f:
            f.write("Talep içinde geçerli bir bağlantı (URL) bulunamadı.")
        return

    existing_articles = load_articles()
    known_urls = set()
    for art in existing_articles:
        known_urls.add(art["link"])
        if art.get("source_url"):
            known_urls.add(art["source_url"])

    new_articles = []
    report_lines = []

    for url in urls:
        if url in known_urls:
            report_lines.append(f"⏭️ Zaten çevrilmiş: {url}")
            continue

        try:
            fetched = fetch_article(url)
            if not fetched:
                report_lines.append(
                    f"❌ Metin çekilemedi (ödeme duvarı veya bot koruması olabilir): {url}"
                )
                continue

            final_url, orig_title, body_md = fetched
            if final_url in known_urls:
                report_lines.append(f"⏭️ Zaten çevrilmiş: {final_url}")
                continue

            raw_output = translate_with_gemini(orig_title, body_md, ARTICLE_SYSTEM_PROMPT)
            if not raw_output:
                raw_output = translate_with_groq(orig_title, body_md, ARTICLE_SYSTEM_PROMPT)

            if not raw_output:
                report_lines.append(f"❌ Çeviri API hatası: {url}")
                continue

            tr_title, tr_html = parse_llm_output(raw_output, orig_title)
            full_title = f"[Tam Makale] {tr_title}"
            tr_html += f'<hr><p><small>Orijinal kaynak: <a href="{final_url}">{final_url}</a></small></p>'

            new_articles.append(
                {
                    "id": f"article:{final_url}",
                    "title": full_title,
                    "original_title": orig_title,
                    "link": final_url,
                    "source_url": url,
                    "published_iso": datetime.now(timezone.utc).isoformat(),
                    "content_html": tr_html,
                }
            )
            known_urls.add(final_url)
            known_urls.add(url)
            report_lines.append(f"✅ Çevrildi ve Feeder'a eklendi: **{full_title}**")
        except Exception as e:
            report_lines.append(f"❌ Hata ({url}): {e}")

    if new_articles:
        combined = new_articles + existing_articles
        combined.sort(key=lambda x: x["published_iso"], reverse=True)
        save_articles(combined)
        build_rss(combined)
        build_picker_html(combined)

    with open("result_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
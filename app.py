"""
🎬 쿠팡 파트너스 쇼츠 자동 생성기
- 네이버 쇼핑 베스트셀러 크롤링
- Gemini API 스크립트 자동 작성 (무료)
- Edge TTS AI 나레이션
- moviepy 영상 합성 → MP4 다운로드
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import asyncio
import nest_asyncio
import edge_tts
import imageio_ffmpeg
import os, re, io, json, time, textwrap, urllib.request, urllib.parse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips

# ── ffmpeg 경로 강제 지정 (Streamlit Cloud 포함) ──────────────
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

# ── asyncio 중첩 허용 (Streamlit 이벤트 루프 충돌 방지) ────────
nest_asyncio.apply()

# ══════════════════════════════════════════════════════════════
# 상수
# ══════════════════════════════════════════════════════════════
VIDEO_W, VIDEO_H = 720, 1280
FPS = 24

CATEGORIES = {
    "🏠 생활/주방": "주방용품 생활용품 인기",
    "💄 뷰티/화장품": "화장품 스킨케어 인기",
    "🍎 식품/건강": "건강식품 영양제 인기",
    "📱 전자/가전": "전자제품 가전 인기",
    "⚽ 스포츠/레저": "스포츠용품 운동 인기",
    "🐾 반려동물": "반려동물 강아지 고양이 인기",
    "👶 유아동": "유아용품 아기 인기",
    "👗 패션/의류": "의류 패션 베스트",
}

ACCENT_COLORS = {
    "hook":     (255, 70,  70),
    "feature1": (65,  140, 255),
    "feature2": (60,  210, 130),
    "feature3": (255, 200, 60),
    "price":    (185, 100, 255),
    "cta":      (255, 70,  70),
}

SECTION_BADGE = {
    "hook":     "🔥 지금 인기",
    "feature1": "✅ 포인트 1",
    "feature2": "✅ 포인트 2",
    "feature3": "✅ 포인트 3",
    "price":    "💰 가격 정보",
    "cta":      "🛒 구매 안내",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9",
}

# ══════════════════════════════════════════════════════════════
# 한국어 폰트
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def _load_font_path() -> str | None:
    candidates = [
        # Streamlit Cloud / Ubuntu – NotoSansCJK (한국어 지원)
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Bold.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        # Nanum (설치된 경우)
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        # Windows / macOS
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def get_font(size: int) -> ImageFont.FreeTypeFont:
    fp = _load_font_path()
    if fp:
        try:
            return ImageFont.truetype(fp, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ══════════════════════════════════════════════════════════════
# 네이버 쇼핑 크롤링
# ══════════════════════════════════════════════════════════════
@st.cache_data(ttl=1800, show_spinner=False)
def search_naver(query: str, n: int = 12) -> list[dict]:
    """네이버 쇼핑 검색 – 리뷰 순 정렬"""
    enc = urllib.parse.quote(query)
    url = (
        f"https://search.shopping.naver.com/search/all"
        f"?query={enc}&sort=review&pagingSize={n}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        products: list[dict] = []

        # ① __NEXT_DATA__ JSON 파싱 (가장 안정적)
        tag = soup.find("script", {"id": "__NEXT_DATA__"})
        if tag:
            try:
                data = json.loads(tag.string)
                items = (
                    data.get("props", {})
                    .get("pageProps", {})
                    .get("initialState", {})
                    .get("products", {})
                    .get("list", [])
                )
                for it in items[:n]:
                    d = it.get("item", it)
                    name = d.get("productTitle") or d.get("name", "")
                    price = d.get("price") or d.get("lprice", 0)
                    img   = d.get("imageUrl") or d.get("image", "")
                    rev   = d.get("reviewCount", "")
                    if img.startswith("//"):
                        img = "https:" + img
                    if name:
                        products.append({
                            "name": str(name)[:55],
                            "price": f"{int(price):,}원" if isinstance(price, (int, float)) and price else str(price),
                            "image_url": img,
                            "review_count": str(rev) if rev else "",
                        })
            except Exception:
                pass

        # ② HTML 파싱 fallback
        if not products:
            items = []
            for sel in [
                "div.product_item__MDtDF",
                "li.product_item",
                "div[class*='product_item']",
                "li[class*='basicList_item']",
            ]:
                items = soup.select(sel)
                if items:
                    break
            for it in items[:n]:
                try:
                    nm  = it.select_one("a[class*='product_link'],span[class*='name'],a[class*='name']")
                    pr  = it.select_one("span[class*='price_num'],em[class*='price'],strong[class*='price']")
                    img = it.select_one("img")
                    rv  = it.select_one("span[class*='review'],em[class*='review']")
                    if nm and img:
                        iu = img.get("src") or img.get("data-src", "")
                        if iu.startswith("//"):
                            iu = "https:" + iu
                        products.append({
                            "name": nm.get_text(strip=True)[:55],
                            "price": pr.get_text(strip=True) if pr else "",
                            "image_url": iu,
                            "review_count": rv.get_text(strip=True) if rv else "",
                        })
                except Exception:
                    continue

        return [p for p in products if p.get("name")]
    except Exception:
        return []


def fetch_image(url: str) -> Image.Image:
    """이미지 URL → PIL Image (실패시 기본 이미지)"""
    try:
        if url:
            r = requests.get(url, headers=HEADERS, timeout=12)
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    # 기본 이미지 (어두운 배경)
    img = Image.new("RGB", (400, 400), (28, 28, 42))
    draw = ImageDraw.Draw(img)
    draw.text((150, 185), "NO IMAGE", fill=(80, 80, 100))
    return img


# ══════════════════════════════════════════════════════════════
# Gemini API – 스크립트 생성
# ══════════════════════════════════════════════════════════════
def generate_script(product: dict, api_key: str) -> dict:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",        # 무료 티어 지원 (소형 모델)
        generation_config=genai.types.GenerationConfig(
            temperature=0.8,
            max_output_tokens=1500,
        ),
    )

    prompt = f"""당신은 쿠팡 파트너스 쇼츠 영상 전문 마케터입니다.

제품 정보:
- 제품명: {product['name']}
- 가격: {product.get('price', '미상')}
- 리뷰 수: {product.get('review_count', '다수')}

30~40초 분량 쇼츠 스크립트를 작성하세요.
조건:
- 각 섹션 나레이션은 4~7초 분량 (약 50~90자 한국어 구어체)
- hook은 궁금증·놀라움 유발, feature는 구체적 혜택 중심
- cta는 "설명란 링크 클릭" 유도
- subtitle은 20자 이내 핵심 자막

JSON만 출력 (마크다운·백틱 없이):
{{
  "title": "유튜브 영상 제목 50자 이내",
  "sections": [
    {{"id":"hook",     "text":"나레이션 텍스트", "subtitle":"자막"}},
    {{"id":"feature1", "text":"나레이션 텍스트", "subtitle":"자막"}},
    {{"id":"feature2", "text":"나레이션 텍스트", "subtitle":"자막"}},
    {{"id":"feature3", "text":"나레이션 텍스트", "subtitle":"자막"}},
    {{"id":"price",    "text":"나레이션 텍스트", "subtitle":"자막"}},
    {{"id":"cta",      "text":"나레이션 텍스트", "subtitle":"자막"}}
  ]
}}"""

    resp = model.generate_content(prompt)
    raw = resp.text.strip()
    # 마크다운 코드블록 제거
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    raw = re.sub(r"^```\s*|\s*```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)


# ══════════════════════════════════════════════════════════════
# Edge TTS – 음성 생성
# ══════════════════════════════════════════════════════════════
def tts(text: str, path: str, voice: str, rate: str = "+8%") -> None:
    async def _run():
        c = edge_tts.Communicate(text, voice, rate=rate)
        await c.save(path)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run())
    finally:
        loop.close()


# ══════════════════════════════════════════════════════════════
# 영상 프레임 생성 (PIL)
# ══════════════════════════════════════════════════════════════
def make_frame(product_img: Image.Image,
               subtitle: str,
               section_id: str) -> np.ndarray:
    """9:16 세로형 프레임 한 장 생성"""
    W, H = VIDEO_W, VIDEO_H
    canvas = Image.new("RGB", (W, H), (18, 18, 28))

    # ── 상품 이미지 (상단 55%) ─────────────────────────────
    img_h = int(H * 0.55)
    ow, oh = product_img.size
    scale = max(W / ow, img_h / oh)
    rw, rh = int(ow * scale), int(oh * scale)
    resized = product_img.resize((rw, rh), Image.LANCZOS)
    left = (rw - W) // 2
    top  = (rh - img_h) // 2
    cropped = resized.crop((left, top, left + W, top + img_h))
    canvas.paste(cropped, (0, 0))

    # ── 이미지 하단 페이드 ────────────────────────────────
    fade_rows = 160
    arr = np.array(canvas)
    bg = np.array([18, 18, 28], dtype=np.float32)
    for i in range(fade_rows):
        alpha = i / fade_rows
        row_y = img_h - fade_rows + i
        if 0 <= row_y < H:
            arr[row_y] = (arr[row_y] * (1 - alpha) + bg * alpha).astype(np.uint8)
    canvas = Image.fromarray(arr)

    draw = ImageDraw.Draw(canvas)
    color = ACCENT_COLORS.get(section_id, (65, 140, 255))

    # ── 구분 라인 ─────────────────────────────────────────
    line_y = img_h + 8
    draw.rectangle([36, line_y, W - 36, line_y + 5], fill=color)

    # ── 섹션 배지 ─────────────────────────────────────────
    badge = SECTION_BADGE.get(section_id, "")
    if badge:
        bf = get_font(30)
        draw.text((36, img_h + 20), badge, fill=color, font=bf)

    # ── 자막 텍스트 ───────────────────────────────────────
    font = get_font(56)
    lines = textwrap.wrap(subtitle, width=14)
    ty = img_h + 72
    for line in lines[:3]:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        tx = (W - tw) // 2
        # 그림자
        draw.text((tx + 3, ty + 3), line, fill=(0, 0, 0),       font=font)
        # 본문
        draw.text((tx,     ty),     line, fill=(255, 255, 255),  font=font)
        ty += 68

    # ── 하단 고정 CTA 바 ─────────────────────────────────
    bar_y = H - 105
    draw.rectangle([0, bar_y, W, H], fill=(25, 25, 38))
    cta_font = get_font(33)
    cta_txt  = "👇 설명란 쿠팡 링크 클릭!"
    cb = draw.textbbox((0, 0), cta_txt, font=cta_font)
    cx = (W - (cb[2] - cb[0])) // 2
    draw.text((cx, bar_y + 32), cta_txt, fill=color, font=cta_font)

    return np.array(canvas)


# ══════════════════════════════════════════════════════════════
# 영상 합성
# ══════════════════════════════════════════════════════════════
def build_video(product: dict,
                script: dict,
                product_img: Image.Image,
                voice: str,
                output_path: str,
                on_progress=None) -> None:
    clips      = []
    temp_audio = []

    sections = script["sections"]
    for i, sec in enumerate(sections):
        if on_progress:
            on_progress(
                0.10 + 0.75 * (i / len(sections)),
                f"섹션 {i+1}/{len(sections)}: '{sec['subtitle']}' 생성 중…"
            )

        # TTS 저장
        ap = f"/tmp/_tts_{i}_{int(time.time()*1000)}.mp3"
        temp_audio.append(ap)
        tts(sec["text"], ap, voice)

        # 오디오 길이 측정
        audio = AudioFileClip(ap)
        dur   = audio.duration + 0.35   # 짧은 여운

        # 프레임 → ImageClip
        frame = make_frame(product_img, sec["subtitle"], sec["id"])
        clip  = ImageClip(frame, duration=dur).set_audio(audio)
        clips.append(clip)

    if on_progress:
        on_progress(0.88, "클립 합치는 중…")

    final = concatenate_videoclips(clips, method="compose")

    if on_progress:
        on_progress(0.92, "MP4 인코딩 중… (잠시 기다려주세요)")

    final.write_videofile(
        output_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="/tmp/_final_audio_tmp.m4a",
        remove_temp=True,
        logger=None,
        threads=4,
        preset="ultrafast",
    )

    # 정리
    final.close()
    for c in clips:
        c.close()
    for f in temp_audio:
        try:
            os.remove(f)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════
# Streamlit UI
# ══════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="쿠팡 파트너스 쇼츠 생성기",
        page_icon="🎬",
        layout="wide",
    )

    st.markdown("""
    <style>
    .stButton>button { border-radius:10px; font-weight:bold; }
    div[data-testid="stExpander"] { border:1px solid #333; border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🎬 쿠팡 파트너스 쇼츠 자동 생성기")
    st.caption("베스트셀러 상품 크롤링 → AI 스크립트 (Gemini) → 나레이션 → MP4 완성")

    # ── 사이드바 ──────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 설정")

        # Streamlit Cloud secrets 우선, 없으면 직접 입력
        _secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if _secret_key:
            api_key = _secret_key
            st.success("🔑 API Key: secrets 적용됨")
        else:
            api_key = st.text_input(
                "🔑 Gemini API Key",
                type="password",
                placeholder="AIza...",
                help="Google AI Studio에서 무료 발급: aistudio.google.com",
            )
            st.caption("키는 세션 내에서만 사용되며 저장되지 않습니다")
            st.markdown(
                "🆓 [무료 키 발급 →](https://aistudio.google.com/app/apikey)",
                unsafe_allow_html=False,
            )

        st.divider()

        VOICES = {
            "선희 (여성, 밝고 자연스러움)":  "ko-KR-SunHiNeural",
            "인준 (남성, 차분함)":            "ko-KR-InJoonNeural",
            "현수 (남성, 활기참)":            "ko-KR-HyunsuNeural",
        }
        voice_label = st.selectbox("🎙️ 나레이션 음성", list(VOICES.keys()))
        voice = VOICES[voice_label]

        st.divider()

        partner_link = st.text_input(
            "🔗 쿠팡 파트너스 링크 (선택)",
            placeholder="https://link.coupang.com/...",
            help="완성 후 설명란에 넣을 링크를 미리 준비하세요"
        )

        st.divider()
        st.info("💡 영상 완성 후 설명란에\n파트너스 링크를 넣으면\n구매 발생 시 수익 발생!")

    if not api_key:
        st.warning("👈 왼쪽 사이드바에서 **Gemini API Key**를 입력해주세요")
        st.info("🆓 **무료 발급:** [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) 접속 → Google 로그인 → 'Create API key' 클릭")
        st.stop()

    # ══════════════════════════════════════════════════════════
    # STEP 1 – 카테고리
    # ══════════════════════════════════════════════════════════
    st.subheader("① 카테고리 선택")

    cat_cols = st.columns(4)
    for i, cat in enumerate(CATEGORIES):
        with cat_cols[i % 4]:
            if st.button(cat, use_container_width=True, key=f"cat_{i}"):
                for k in ["products", "selected_product", "script",
                          "video_done", "video_path"]:
                    st.session_state.pop(k, None)
                st.session_state["category"] = cat

    if "category" not in st.session_state:
        st.stop()

    st.success(f"선택된 카테고리: **{st.session_state['category']}**")

    # ══════════════════════════════════════════════════════════
    # STEP 2 – 상품 선택
    # ══════════════════════════════════════════════════════════
    st.subheader("② 인기 상품 선택")

    if "products" not in st.session_state:
        q = CATEGORIES[st.session_state["category"]]
        with st.spinner("🔍 네이버 쇼핑에서 인기 상품 검색 중…"):
            st.session_state["products"] = search_naver(q, n=12)

    products = st.session_state.get("products", [])

    # ── 크롤링 실패 → 수동 입력 ──────────────────────────────
    if not products:
        st.warning("⚠️ 자동 검색에 실패했습니다. 상품 정보를 직접 입력해주세요.")
        with st.form("manual_form"):
            m_name  = st.text_input("제품명 *")
            m_price = st.text_input("가격 (예: 29,900원)")
            m_img   = st.text_input("이미지 URL (선택)")
            m_rev   = st.text_input("리뷰 수 (예: 3,421개)")
            if st.form_submit_button("이 상품으로 영상 만들기", type="primary"):
                if m_name:
                    st.session_state["selected_product"] = {
                        "name": m_name, "price": m_price,
                        "image_url": m_img, "review_count": m_rev,
                    }
                    st.session_state.pop("script", None)
                    st.session_state.pop("video_done", None)
    else:
        st.write(f"✅ **{len(products)}개** 인기 상품을 찾았습니다")

        labels = [
            f"{p['name'][:38]}{'…' if len(p['name'])>38 else ''}"
            f"  ({p['price']})" + (f"  ⭐{p['review_count']}" if p.get("review_count") else "")
            for p in products
        ]
        idx = st.selectbox("상품을 선택하세요", range(len(products)),
                           format_func=lambda x: labels[x])
        sel = products[idx]

        # 미리보기
        c1, c2 = st.columns([1, 3])
        with c1:
            if sel.get("image_url"):
                try:
                    st.image(sel["image_url"], width=140)
                except Exception:
                    pass
        with c2:
            st.markdown(f"**📦 제품명:** {sel['name']}")
            st.markdown(f"**💰 가격:** {sel['price']}")
            if sel.get("review_count"):
                st.markdown(f"**⭐ 리뷰:** {sel['review_count']}")

        if st.button("🎬 이 상품으로 쇼츠 만들기", type="primary", use_container_width=True):
            st.session_state["selected_product"] = sel
            for k in ["script", "video_done", "video_path"]:
                st.session_state.pop(k, None)

    if "selected_product" not in st.session_state:
        st.stop()

    # ══════════════════════════════════════════════════════════
    # STEP 3 – 영상 생성
    # ══════════════════════════════════════════════════════════
    product = st.session_state["selected_product"]
    st.subheader("③ 쇼츠 영상 생성")
    st.info(f"🛍️ 선택 상품: **{product['name']}**")

    col_go, col_back = st.columns(2)
    with col_go:
        go = st.button("🚀 영상 생성 시작", type="primary",
                       use_container_width=True,
                       disabled=st.session_state.get("video_done", False))
    with col_back:
        if st.button("↩️ 상품 다시 선택", use_container_width=True):
            for k in ["selected_product", "script", "video_done", "video_path"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── 생성 프로세스 ─────────────────────────────────────────
    if go and not st.session_state.get("video_done"):
        bar    = st.progress(0.0)
        status = st.empty()
        try:
            # 1. 스크립트
            status.text("📝 Gemini가 스크립트 작성 중…")
            bar.progress(0.05)
            script = generate_script(product, api_key)
            st.session_state["script"] = script

            # 2. 이미지
            status.text("🖼️ 제품 이미지 다운로드 중…")
            bar.progress(0.10)
            pimg = fetch_image(product.get("image_url", ""))

            # 3. 영상
            out = f"/tmp/shorts_{int(time.time())}.mp4"
            def _progress(pct, msg):
                bar.progress(pct)
                status.text(f"🎬 {msg}")

            build_video(product, script, pimg, voice, out, _progress)

            bar.progress(1.0)
            status.text("✅ 완성!")
            st.session_state["video_path"] = out
            st.session_state["video_done"] = True
            st.rerun()

        except json.JSONDecodeError as e:
            st.error(f"스크립트 JSON 파싱 오류 – Gemini 응답을 확인해주세요: {e}")
        except Exception as e:
            st.error(f"오류 발생: {e}")
            import traceback
            with st.expander("상세 오류 보기"):
                st.code(traceback.format_exc())

    # ── 완성 화면 ─────────────────────────────────────────────
    if st.session_state.get("video_done"):
        st.success("🎉 쇼츠 영상 완성!")

        if "script" in st.session_state:
            scr = st.session_state["script"]
            with st.expander("📋 생성된 스크립트 보기"):
                st.markdown(f"**📌 영상 제목:** {scr.get('title','')}")
                st.divider()
                for s in scr.get("sections", []):
                    st.markdown(f"**[{s['id']}]** {s['text']}")
                    st.caption(f"자막: {s['subtitle']}")

        vp = st.session_state.get("video_path", "")
        if vp and os.path.exists(vp):
            st.video(vp)
            with open(vp, "rb") as f:
                vbytes = f.read()
            fname = re.sub(r"[^\w가-힣]", "_", product["name"][:20]) + ".mp4"
            st.download_button(
                "⬇️ MP4 파일 다운로드",
                data=vbytes,
                file_name=fname,
                mime="video/mp4",
                use_container_width=True,
            )

        if partner_link:
            st.markdown("---")
            st.markdown("**📌 유튜브 영상 설명란에 넣을 내용 (복사하세요):**")
            scr_title = st.session_state.get("script", {}).get("title", product["name"])
            st.code(
                f"{scr_title}\n\n"
                f"쿠팡에서 바로 구매하기 👇\n"
                f"{partner_link}\n\n"
                f"※ 이 링크는 쿠팡 파트너스 링크로,\n"
                f"   구매 시 소정의 수수료를 받을 수 있습니다.",
                language=None,
            )

        st.divider()
        if st.button("🔄 새 영상 만들기", use_container_width=True):
            for k in ["video_done", "video_path", "script",
                      "selected_product", "products", "category"]:
                st.session_state.pop(k, None)
            st.rerun()


if __name__ == "__main__":
    main()
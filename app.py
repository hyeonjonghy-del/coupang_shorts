"""
🎬 쿠팡 파트너스 쇼츠 자동 생성기 v2
- 네이버 쇼핑 크롤링 or 수동 입력
- Gemini API 스크립트 자동 작성
- edge-tts 자연스러운 한국어 나레이션 (fallback: gTTS)
- 제품 이미지 업로드 + Ken Burns 줌 효과
- moviepy MP4 다운로드
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import subprocess, asyncio, nest_asyncio
import imageio_ffmpeg
import os, re, io, json, time, textwrap, urllib.parse

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import VideoClip, AudioFileClip, concatenate_videoclips

os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
nest_asyncio.apply()

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
    "cta":      (255, 120, 50),
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

VOICES = {
    "선희 (여성, 밝고 자연스러움)": "ko-KR-SunHiNeural",
    "인준 (남성, 차분함)":          "ko-KR-InJoonNeural",
    "현수 (남성, 활기참)":          "ko-KR-HyunsuNeural",
}

# ── 폰트 ──────────────────────────────────────────────────────
@st.cache_resource
def _load_font_path():
    for p in [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Bold.otf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    ]:
        if os.path.exists(p):
            return p
    return None

def get_font(size):
    fp = _load_font_path()
    if fp:
        try: return ImageFont.truetype(fp, size)
        except: pass
    return ImageFont.load_default()

# ── 네이버 쇼핑 크롤링 ───────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def search_naver(query, n=12):
    enc = urllib.parse.quote(query)
    url = f"https://search.shopping.naver.com/search/all?query={enc}&sort=review&pagingSize={n}"
    try:
        soup = BeautifulSoup(requests.get(url, headers=HEADERS, timeout=15).text, "html.parser")
        products = []
        tag = soup.find("script", {"id": "__NEXT_DATA__"})
        if tag:
            try:
                items = (json.loads(tag.string).get("props",{}).get("pageProps",{})
                         .get("initialState",{}).get("products",{}).get("list",[]))
                for it in items[:n]:
                    d = it.get("item", it)
                    name  = d.get("productTitle") or d.get("name","")
                    price = d.get("price") or d.get("lprice", 0)
                    img   = d.get("imageUrl") or d.get("image","")
                    rev   = d.get("reviewCount","")
                    if img.startswith("//"): img = "https:" + img
                    if name:
                        products.append({
                            "name": str(name)[:55],
                            "price": f"{int(price):,}원" if isinstance(price,(int,float)) and price else str(price),
                            "image_url": img,
                            "review_count": str(rev) if rev else "",
                        })
            except: pass
        return [p for p in products if p.get("name")]
    except:
        return []

# ── Gemini REST API ───────────────────────────────────────────
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

def _get_best_model(api_key):
    preferred = [
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-pro-exp-03-25",
        "gemini-1.5-flash-002", "gemini-1.5-flash-001",
        "gemini-1.5-pro-002",   "gemini-1.5-pro-001",
    ]
    try:
        data = requests.get(f"{GEMINI_BASE}?key={api_key}", timeout=10).json()
        available = [m["name"].replace("models/","")
                     for m in data.get("models",[])
                     if "generateContent" in m.get("supportedGenerationMethods",[])]
        avset = set(available)
        for p in preferred:
            if p in avset: return p
        if available: return available[0]
    except: pass
    return preferred[0]

def generate_script(product, api_key):
    model = _get_best_model(api_key)
    prompt = f"""당신은 쿠팡 파트너스 쇼츠 영상 전문 마케터입니다.

제품 정보:
- 제품명: {product['name']}
- 가격: {product.get('price','미상')}
- 리뷰 수: {product.get('review_count','다수')}

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
    {{"id":"hook",     "text":"나레이션", "subtitle":"자막"}},
    {{"id":"feature1", "text":"나레이션", "subtitle":"자막"}},
    {{"id":"feature2", "text":"나레이션", "subtitle":"자막"}},
    {{"id":"feature3", "text":"나레이션", "subtitle":"자막"}},
    {{"id":"price",    "text":"나레이션", "subtitle":"자막"}},
    {{"id":"cta",      "text":"나레이션", "subtitle":"자막"}}
  ]
}}"""
    resp = requests.post(f"{GEMINI_BASE}/{model}:generateContent?key={api_key}",
                         json={"contents":[{"parts":[{"text":prompt}]}]}, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    raw = re.sub(r"^```json\s*|\s*```$","",raw,flags=re.MULTILINE).strip()
    raw = re.sub(r"^```\s*|\s*```$","",raw,flags=re.MULTILINE).strip()
    return json.loads(raw)

# ── TTS (edge-tts CLI → gTTS fallback) ───────────────────────
def tts(text, path, voice="ko-KR-SunHiNeural", rate="+8%"):
    try:
        r = subprocess.run(
            ["edge-tts","--voice",voice,"--rate",rate,"--text",text,"--write-media",path],
            capture_output=True, timeout=45)
        if r.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 100:
            return
    except Exception:
        pass
    # fallback
    from gtts import gTTS
    gTTS(text=text, lang="ko", slow=False).save(path)
    if not os.path.exists(path) or os.path.getsize(path) < 100:
        raise RuntimeError(f"TTS 생성 실패: {path}")

# ── Ken Burns 프레임 함수 ─────────────────────────────────────
def make_frame_func(product_img, subtitle, section_id, duration):
    W, H   = VIDEO_W, VIDEO_H
    img_h  = int(H * 0.62)
    color  = ACCENT_COLORS.get(section_id, (65,140,255))
    badge  = SECTION_BADGE.get(section_id,"")

    # 최대 줌 기준으로 이미지 미리 크게 리사이즈
    ow, oh   = product_img.size
    base_sc  = max(W/ow, img_h/oh) * 1.15
    rw, rh   = int(ow*base_sc), int(oh*base_sc)
    big      = product_img.resize((rw,rh), Image.LANCZOS)

    f_sub = get_font(58); f_bdg = get_font(30); f_cta = get_font(32)
    lines = textwrap.wrap(subtitle, width=13)

    def make(t):
        prog   = t / max(duration, 0.01)
        zoom   = 1.0 + 0.12 * prog          # 1.0→1.12 줌인
        cw, ch = int(W/zoom), int(img_h/zoom)
        left   = (rw-cw)//2; top = (rh-ch)//2
        crop   = big.crop((left,top,left+cw,top+ch)).resize((W,img_h),Image.LANCZOS)

        canvas = Image.new("RGB",(W,H),(18,18,28))
        canvas.paste(crop,(0,0))

        # 페이드
        arr = np.array(canvas); bg = np.array([18,18,28],dtype=np.float32)
        fade = 180
        for i in range(fade):
            ry = img_h-fade+i
            if 0<=ry<H:
                arr[ry] = (arr[ry]*(1-i/fade)+bg*(i/fade)).astype(np.uint8)
        canvas = Image.fromarray(arr)
        draw   = ImageDraw.Draw(canvas)

        draw.rectangle([36,img_h+6,W-36,img_h+11], fill=color)
        if badge: draw.text((36,img_h+18), badge, fill=color, font=f_bdg)

        ty = img_h+65
        for line in lines[:3]:
            bb = draw.textbbox((0,0),line,font=f_sub)
            tx = (W-(bb[2]-bb[0]))//2
            draw.text((tx+3,ty+3),line,fill=(0,0,0),font=f_sub)
            draw.text((tx,ty),line,fill=(255,255,255),font=f_sub)
            ty += 70

        bar_y = H-100
        draw.rectangle([0,bar_y,W,H],fill=(20,20,34))
        cta = "👇 설명란 쿠팡 링크 클릭!"
        cb  = draw.textbbox((0,0),cta,font=f_cta)
        draw.text(((W-(cb[2]-cb[0]))//2, bar_y+30), cta, fill=color, font=f_cta)

        return np.array(canvas)
    return make

# ── 영상 합성 ─────────────────────────────────────────────────
def build_video(product, script, product_img, voice, output_path, on_progress=None):
    clips, tmps = [], []
    sections    = script["sections"]
    for i, sec in enumerate(sections):
        if on_progress:
            on_progress(0.10+0.75*(i/len(sections)),
                        f"섹션 {i+1}/{len(sections)}: '{sec['subtitle']}' 생성 중…")
        ap = f"/tmp/_tts_{i}_{int(time.time()*1000)}.mp3"
        tmps.append(ap)
        tts(sec["text"], ap, voice)
        audio = AudioFileClip(ap)
        dur   = audio.duration + 0.4
        clip  = VideoClip(make_frame_func(product_img,sec["subtitle"],sec["id"],dur),
                          duration=dur).set_fps(FPS).set_audio(audio)
        clips.append(clip)

    if on_progress: on_progress(0.88,"클립 합치는 중…")
    final = concatenate_videoclips(clips, method="compose")
    if on_progress: on_progress(0.92,"MP4 인코딩 중…")
    final.write_videofile(output_path, fps=FPS, codec="libx264", audio_codec="aac",
                          temp_audiofile="/tmp/_fa.m4a", remove_temp=True,
                          logger=None, threads=2, preset="ultrafast")
    final.close()
    for c in clips: c.close()
    for f in tmps:
        try: os.remove(f)
        except: pass

# ── Streamlit UI ──────────────────────────────────────────────
def main():
    st.set_page_config(page_title="쿠팡 파트너스 쇼츠 생성기", page_icon="🎬", layout="wide")
    st.markdown("<style>.stButton>button{border-radius:10px;font-weight:bold;}</style>",
                unsafe_allow_html=True)
    st.title("🎬 쿠팡 파트너스 쇼츠 자동 생성기")
    st.caption("제품 이미지 업로드 → AI 스크립트 → 자연스러운 나레이션 → MP4")

    with st.sidebar:
        st.header("⚙️ 설정")
        _s = st.secrets.get("GEMINI_API_KEY","")
        if _s:
            api_key = _s; st.success("🔑 secrets 적용됨")
        else:
            api_key = st.text_input("🔑 Gemini API Key", type="password", placeholder="AIza...")
        st.divider()
        voice_label = st.selectbox("🎙️ 나레이션 음성", list(VOICES.keys()))
        voice = VOICES[voice_label]
        st.divider()
        partner_link = st.text_input("🔗 쿠팡 파트너스 링크 (선택)",
                                     placeholder="https://link.coupang.com/...")
        st.divider()
        st.info("💡 영상 완성 후 설명란에\n파트너스 링크를 넣으면\n구매 시 수익 발생!")

    if not api_key:
        st.warning("👈 Gemini API Key를 입력해주세요"); st.stop()

    # ① 카테고리
    st.subheader("① 카테고리 선택")
    cols = st.columns(4)
    for i, cat in enumerate(CATEGORIES):
        with cols[i%4]:
            if st.button(cat, use_container_width=True, key=f"cat_{i}"):
                for k in ["products","selected_product","script","video_done","video_path"]:
                    st.session_state.pop(k,None)
                st.session_state["category"] = cat
    if "category" not in st.session_state: st.stop()
    st.success(f"선택: **{st.session_state['category']}**")

    # ② 상품 + 이미지
    st.subheader("② 상품 정보 & 제품 이미지")
    if "products" not in st.session_state:
        q = CATEGORIES[st.session_state["category"]]
        with st.spinner("🔍 네이버 쇼핑 검색 중…"):
            st.session_state["products"] = search_naver(q)

    products = st.session_state.get("products",[])
    col_info, col_img = st.columns([3,2])

    with col_info:
        if not products:
            st.warning("⚠️ 자동 검색 실패 – 직접 입력")
            with st.form("mf"):
                mn = st.text_input("제품명 *")
                mp = st.text_input("가격")
                mr = st.text_input("리뷰 수")
                if st.form_submit_button("선택", type="primary") and mn:
                    st.session_state["selected_product"] = {
                        "name":mn,"price":mp,"image_url":"","review_count":mr}
                    for k in ["script","video_done","video_path"]: st.session_state.pop(k,None)
        else:
            st.write(f"✅ **{len(products)}개** 인기 상품")
            labels = [f"{p['name'][:35]}  ({p['price']})" + (f"  ⭐{p['review_count']}" if p.get("review_count") else "") for p in products]
            idx = st.selectbox("상품 선택", range(len(products)), format_func=lambda x:labels[x])
            sel = products[idx]
            st.markdown(f"**📦** {sel['name']}  \n**💰** {sel['price']}")
            if st.button("🎬 이 상품으로 진행", type="primary", use_container_width=True):
                st.session_state["selected_product"] = sel
                for k in ["script","video_done","video_path"]: st.session_state.pop(k,None)

    with col_img:
        st.markdown("**📸 제품 이미지 업로드**")
        st.caption("쿠팡/네이버 상품 이미지를 저장해서 올려주세요")
        uploaded = st.file_uploader("이미지", type=["jpg","jpeg","png","webp"],
                                    label_visibility="collapsed")
        if uploaded:
            st.session_state["uploaded_img"] = uploaded.read()
        if "uploaded_img" in st.session_state:
            st.image(st.session_state["uploaded_img"], use_column_width=True)

    if "selected_product" not in st.session_state: st.stop()

    # ③ 영상 생성
    product = st.session_state["selected_product"]
    st.subheader("③ 쇼츠 영상 생성")
    st.info(f"🛍️ **{product['name']}**")
    if "uploaded_img" not in st.session_state:
        st.warning("⬆️ 위에서 제품 이미지를 업로드해주세요!")

    cg, cb = st.columns(2)
    with cg:
        go = st.button("🚀 영상 생성 시작", type="primary", use_container_width=True,
                       disabled=st.session_state.get("video_done",False))
    with cb:
        if st.button("↩️ 다시 선택", use_container_width=True):
            for k in ["selected_product","script","video_done","video_path"]:
                st.session_state.pop(k,None)
            st.rerun()

    if go and not st.session_state.get("video_done"):
        bar = st.progress(0.0); status = st.empty()
        try:
            status.text("📝 Gemini 스크립트 작성 중…"); bar.progress(0.05)
            script = generate_script(product, api_key)
            st.session_state["script"] = script

            status.text("🖼️ 이미지 준비 중…"); bar.progress(0.10)
            if "uploaded_img" in st.session_state:
                pimg = Image.open(io.BytesIO(st.session_state["uploaded_img"])).convert("RGB")
            else:
                pimg = Image.new("RGB",(720,720),(30,30,50))

            out = f"/tmp/shorts_{int(time.time())}.mp4"
            build_video(product, script, pimg, voice, out,
                        lambda pct,msg: (bar.progress(pct), status.text(f"🎬 {msg}")))
            bar.progress(1.0); status.text("✅ 완성!")
            st.session_state.update({"video_path":out,"video_done":True})
            st.rerun()
        except json.JSONDecodeError as e:
            st.error(f"스크립트 JSON 오류: {e}")
        except Exception as e:
            st.error(f"오류: {e}")
            import traceback
            with st.expander("상세 오류"): st.code(traceback.format_exc())

    if st.session_state.get("video_done"):
        st.success("🎉 쇼츠 영상 완성!")
        scr = st.session_state.get("script",{})
        with st.expander("📋 스크립트 보기"):
            st.markdown(f"**📌** {scr.get('title','')}")
            for s in scr.get("sections",[]):
                st.markdown(f"**[{s['id']}]** {s['text']}")
                st.caption(f"자막: {s['subtitle']}")

        vp = st.session_state.get("video_path","")
        if vp and os.path.exists(vp):
            st.video(vp)
            with open(vp,"rb") as f: vb = f.read()
            st.download_button("⬇️ MP4 다운로드", data=vb,
                               file_name=re.sub(r"[^\w가-힣]","_",product["name"][:20])+".mp4",
                               mime="video/mp4", use_container_width=True)
        if partner_link:
            st.markdown("---\n**📌 유튜브 설명란 내용:**")
            st.code(f"{scr.get('title',product['name'])}\n\n쿠팡에서 바로 구매하기 👇\n{partner_link}\n\n※ 쿠팡 파트너스 링크로 구매 시 소정의 수수료를 받을 수 있습니다.", language=None)

        st.divider()
        if st.button("🔄 새 영상 만들기", use_container_width=True):
            for k in ["video_done","video_path","script","selected_product","products","category","uploaded_img"]:
                st.session_state.pop(k,None)
            st.rerun()

if __name__ == "__main__":
    main()
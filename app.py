import streamlit as st
import google.generativeai as genai
import json
import re
from datetime import datetime

st.set_page_config(
    page_title="역사 쇼츠 메이커",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

.hero-title {
    font-size: 2.4rem; font-weight: 900;
    background: linear-gradient(135deg, #ff6b35, #f7c59f, #fffbe6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub { color: #888; font-size: 0.95rem; margin-bottom: 2rem; }

.scene-card {
    background: #13131f; border: 1px solid #2a2a3f;
    border-left: 3px solid #ff6b35; border-radius: 8px;
    padding: 1rem 1.2rem; margin-bottom: 0.8rem;
}
.scene-num { color: #ff6b35; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.1em; }
.scene-text { font-size: 1rem; color: #e8e8f0; margin: 0.3rem 0; line-height: 1.6; }
.scene-img { color: #888; font-size: 0.82rem; }
.scene-dur { color: #f7c59f; font-size: 0.8rem; font-weight: 700; }

.narr-box {
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    padding: 1.2rem; font-size: 0.95rem; line-height: 1.9;
    color: #c9d1d9; white-space: pre-wrap;
}
.tag-pill {
    display: inline-block; background: #1a1a2e; border: 1px solid #2a2a4a;
    border-radius: 20px; padding: 0.25rem 0.7rem;
    font-size: 0.8rem; color: #9090c0; margin: 0.15rem;
}
.stat-box {
    background: #13131f; border: 1px solid #2a2a3f;
    border-radius: 8px; padding: 0.8rem 1rem; text-align: center;
}
.stat-val { font-size: 1.6rem; font-weight: 900; color: #ff6b35; }
.stat-label { font-size: 0.75rem; color: #666; }

.vrew-box {
    background: #0f2027; border: 2px solid #1a6b5a;
    border-radius: 10px; padding: 1.2rem; margin: 1rem 0;
}
.vrew-title { color: #2ecc71; font-weight: 700; font-size: 1rem; margin-bottom: 0.5rem; }
.free-badge {
    display: inline-block; background: #2ecc71; color: #000;
    font-size: 0.7rem; font-weight: 900; padding: 0.1rem 0.5rem;
    border-radius: 20px; margin-left: 0.5rem; vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ─── 추천 주제 ────────────────────────────────────────────
SUGGESTED_TOPICS = [
    ("💰 조선이 망한 진짜 이유는 '돈'이었다", "조선 경제 붕괴와 현대 경제 위기 비교"),
    ("🌷 튤립 버블 — 400년 전 코인 폭락과 똑같다", "17세기 네덜란드 튤립 투기와 현대 암호화폐"),
    ("🏯 임진왜란이 일본을 부자로 만든 이유", "도자기 기술 약탈과 산업화의 관계"),
    ("🗺️ 신라가 사막 나라와 교역한 방법", "실크로드와 신라의 국제 무역"),
    ("⚔️ 로마가 멸망한 진짜 원인은 세금이었다", "세금 과부담과 제국 붕괴의 경제학"),
    ("🏦 조선 화폐 개혁이 실패한 이유", "상평통보와 현대 화폐 정책 비교"),
    ("🚢 대항해시대가 만든 최초의 글로벌 경제", "콜럼버스 이후 세계 무역 구조 변화"),
    ("📉 대공황 — 1929년과 2008년은 왜 똑같이 반복됐나", "역사 속 금융위기 패턴 분석"),
    ("🛢️ 석유 패권 — 중동이 부자가 된 진짜 이유", "1973년 오일쇼크와 현재 에너지 전쟁"),
    ("🏗️ 일제강점기 조선 경제는 성장했나 수탈당했나", "식민지 근대화론 vs 수탈론"),
]

# ─── API 키 로드 ──────────────────────────────────────────
def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None

# ─── 헤더 ─────────────────────────────────────────────────
st.markdown('<div class="hero-title">🎬 역사 쇼츠 메이커</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">역사 + 경제 연결 콘텐츠 · 1분 쇼츠 자동 생성 · Powered by Gemini (무료)</div>', unsafe_allow_html=True)

api_key = get_api_key()
if not api_key:
    st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Cloud Secrets에 API 키를 입력해주세요.")
    with st.expander("🔑 Gemini API 키 무료 발급 방법"):
        st.markdown("""
1. **https://aistudio.google.com** 접속 (구글 계정 로그인)
2. **`Get API key`** 클릭
3. **`Create API key`** 클릭
4. 생성된 키 복사 (`AIza...` 형태)
5. Streamlit Cloud → Settings → Secrets 에 아래 형식으로 입력:
```toml
GEMINI_API_KEY = "AIza..."
```
> ✅ 완전 무료! 크레딧 충전 불필요
        """)
    st.stop()

# ─── 입력 섹션 ────────────────────────────────────────────
col_left, col_right = st.columns([1.2, 1], gap="large")

with col_left:
    st.markdown("#### ✏️ 주제 설정")
    topic_mode = st.radio("입력 방식", ["추천 주제 선택", "직접 입력"], horizontal=True)

    if topic_mode == "추천 주제 선택":
        selected = st.selectbox(
            "주제 선택",
            options=range(len(SUGGESTED_TOPICS)),
            format_func=lambda i: SUGGESTED_TOPICS[i][0],
        )
        topic = SUGGESTED_TOPICS[selected][0]
        topic_hint = SUGGESTED_TOPICS[selected][1]
        st.caption(f"💡 {topic_hint}")
    else:
        topic = st.text_input("주제를 직접 입력하세요", placeholder="예: 고려청자가 세계 최고가 된 경제적 이유")
        topic_hint = ""

    target = st.selectbox("타겟 시청자", ["일반 대중 (쉽고 흥미롭게)", "경제·투자 관심층 (심층 분석)", "역사 마니아 (전문적 서술)"])
    duration = st.select_slider("영상 길이", options=["45초", "60초", "75초", "90초"], value="60초")
    tone = st.selectbox("나레이션 톤", ["드라마틱·몰입감 있게", "차분·다큐 스타일", "친근·대화체"])

with col_right:
    st.markdown("#### 📋 생성 옵션")
    gen_script = st.checkbox("📝 장면별 스크립트", value=True)
    gen_narration = st.checkbox("🎙️ 나레이션 전문 (Vrew용)", value=True)
    gen_tags = st.checkbox("#️⃣ 해시태그", value=True)
    gen_title = st.checkbox("📌 제목 후보 3개", value=True)

    st.markdown("---")
    st.markdown("#### 🆓 무료 제작 워크플로우")
    st.markdown("""
<div class="vrew-box">
<div class="vrew-title">📋 Vrew 활용법 <span class="free-badge">완전 무료</span></div>

1. 여기서 스크립트 생성<br>
2. <b>나레이션 전문</b> 복사<br>
3. <a href="https://vrew.ai" target="_blank">vrew.ai</a> → 새 프로젝트<br>
4. <b>텍스트로 비디오 만들기</b> 선택<br>
5. 스크립트 붙여넣기 → AI 음성 선택<br>
6. 자동으로 이미지+자막+음성 완성!<br>
7. 유튜브 업로드 🚀
</div>
""", unsafe_allow_html=True)

st.markdown("---")
generate_btn = st.button("🚀 쇼츠 콘텐츠 생성하기", type="primary", use_container_width=True)

# ─── 생성 로직 ────────────────────────────────────────────
if generate_btn and topic:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    sec_map = {"45초": 45, "60초": 60, "75초": 75, "90초": 90}
    secs = sec_map[duration]
    scene_count = secs // 10

    prompt = f"""당신은 한국 유튜브 역사 쇼츠 콘텐츠 전문 작가입니다.
역사와 경제·투자를 연결하는 차별화된 콘텐츠를 만드세요.

주제: {topic}
타겟: {target}
영상 길이: {duration} (총 {secs}초, {scene_count}개 장면)
나레이션 톤: {tone}

반드시 아래 JSON 형식으로만 응답하세요 (마크다운 코드블록 없이 순수 JSON만):
{{
  "title_candidates": ["제목1", "제목2", "제목3"],
  "hook": "첫 3초 후킹 문장 (시청자가 멈추게 만드는 강렬한 한 줄)",
  "scenes": [
    {{
      "scene_no": 1,
      "duration_sec": 10,
      "narration": "나레이션 텍스트",
      "subtitle": "화면 자막 (10자 이내)",
      "image_keyword_ko": "이미지 검색어(한국어)",
      "image_keyword_en": "image search keyword in English"
    }}
  ],
  "full_narration": "전체 나레이션 하나의 흐름으로 (Vrew에 바로 붙여넣을 수 있게)",
  "vrew_keywords": "Vrew 키워드 검색용 핵심단어 3~5개 (쉼표 구분)",
  "hashtags": ["태그1","태그2","태그3","태그4","태그5","태그6","태그7","태그8","태그9","태그10"],
  "production_tips": "Vrew로 이 영상 만들 때 추천 배경 스타일 또는 주의사항"
}}"""

    with st.spinner("🎬 Gemini가 콘텐츠를 생성하는 중..."):
        try:
            response = model.generate_content(prompt)
            raw = response.text.strip()
            raw_clean = re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()
            data = json.loads(raw_clean)

            st.success("✅ 생성 완료!")
            st.markdown("---")

            # 제목 후보
            if gen_title and "title_candidates" in data:
                st.markdown("### 📌 제목 후보")
                for i, t in enumerate(data["title_candidates"], 1):
                    st.markdown(f"**{i}.** {t}")

            # 후킹 문장
            if "hook" in data:
                st.markdown("### ⚡ 오프닝 후킹")
                st.markdown(f"> **{data['hook']}**")

            # 통계
            if "scenes" in data:
                c1, c2, c3, c4 = st.columns(4)
                total_sec = sum(s.get("duration_sec", 10) for s in data["scenes"])
                narr_len = len(data.get("full_narration", ""))
                for col, val, label in zip(
                    [c1, c2, c3, c4],
                    [len(data["scenes"]), f"{total_sec}s", narr_len, len(data.get("hashtags", []))],
                    ["장면 수", "총 길이", "나레이션 글자 수", "해시태그"]
                ):
                    with col:
                        st.markdown(f'<div class="stat-box"><div class="stat-val">{val}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)
                st.markdown("")

            # 장면별 스크립트
            if gen_script and "scenes" in data:
                st.markdown("### 🎬 장면별 스크립트")
                for scene in data["scenes"]:
                    img_en = scene.get("image_keyword_en", "")
                    pexels_url = f"https://www.pexels.com/search/{img_en.replace(' ', '%20')}/" if img_en else ""
                    img_line = f'🔍 {scene.get("image_keyword_ko","")} | <a href="{pexels_url}" target="_blank">Pexels 검색 →</a>' if pexels_url else f'🔍 {scene.get("image_keyword_ko","")}'
                    st.markdown(f"""
<div class="scene-card">
  <div class="scene-num">SCENE {scene.get('scene_no','')} &nbsp;·&nbsp; <span class="scene-dur">⏱ {scene.get('duration_sec',10)}초</span></div>
  <div class="scene-text">🎙️ {scene.get('narration','')}</div>
  <div class="scene-text" style="color:#f7c59f;font-size:0.9rem;">📺 자막: <strong>{scene.get('subtitle','')}</strong></div>
  <div class="scene-img">{img_line}</div>
</div>""", unsafe_allow_html=True)

            # 나레이션 전문 (Vrew용)
            if gen_narration and "full_narration" in data:
                st.markdown("### 🎙️ 나레이션 전문")
                st.markdown("""
<div class="vrew-box">
<div class="vrew-title">📋 Vrew 사용법</div>
① 아래 텍스트 전체 복사 → ② vrew.ai 접속 → ③ <b>텍스트로 비디오 만들기</b> → ④ 붙여넣기 → ⑤ AI 음성 선택 → 완성!
</div>
""", unsafe_allow_html=True)
                st.markdown('<div class="narr-box">' + data["full_narration"] + '</div>', unsafe_allow_html=True)
                st.code(data["full_narration"], language=None)

            # Vrew 키워드
            if "vrew_keywords" in data:
                st.markdown("### 🔍 Vrew 이미지 검색 키워드")
                st.info(f"**{data['vrew_keywords']}**\n\nVrew 편집기에서 이 키워드로 배경 이미지를 검색하세요!")

            # 해시태그
            if gen_tags and "hashtags" in data:
                st.markdown("### #️⃣ 해시태그")
                tags_html = " ".join([f'<span class="tag-pill">#{t}</span>' for t in data["hashtags"]])
                st.markdown(tags_html, unsafe_allow_html=True)
                st.code(" ".join([f"#{t}" for t in data["hashtags"]]), language=None)

            # 제작 팁
            if "production_tips" in data:
                st.markdown("### 💡 Vrew 제작 팁")
                st.info(data["production_tips"])

            # JSON 저장
            st.markdown("---")
            now = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "💾 전체 데이터 저장 (JSON)",
                data=json.dumps(data, ensure_ascii=False, indent=2),
                file_name=f"shorts_{now}.json",
                mime="application/json"
            )

        except json.JSONDecodeError as e:
            st.error(f"JSON 파싱 오류: {e}")
            st.code(raw)
        except Exception as e:
            st.error(f"오류: {e}")

elif generate_btn and not topic:
    st.warning("주제를 입력해주세요!")

# 푸터
with st.expander("📖 완전 무료 제작 가이드"):
    st.markdown("""
| 단계 | 도구 | 비용 | 링크 |
|------|------|------|------|
| 1️⃣ 스크립트 생성 | 이 앱 (Gemini) | **무료** | — |
| 2️⃣ 영상+음성+자막 | **Vrew** | **무료** | [vrew.ai](https://vrew.ai) |
| 3️⃣ 추가 편집 | CapCut | **무료** | [capcut.com](https://capcut.com) |
| 4️⃣ 이미지 소스 | Pexels | **무료** | [pexels.com](https://pexels.com) |
| 5️⃣ BGM | YT 오디오 라이브러리 | **무료** | [studio.youtube.com](https://studio.youtube.com) |
| 6️⃣ 업로드 | YouTube Studio | **무료** | — |

### 🔑 Gemini API 키 무료 발급
1. [aistudio.google.com](https://aistudio.google.com) 접속 (구글 계정)
2. `Get API key` → `Create API key`
3. Streamlit Cloud Secrets에 입력:
```toml
GEMINI_API_KEY = "AIza..."
```
""")

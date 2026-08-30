PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&display=swap');
:root { --bg-a: #f4f8ff; --bg-b: #eef9f2; --ink: #0f2438; --muted: #4f6172; --primary: #0f6adf; --primary-2: #004fc7; --card: #ffffff; --line: #d9e4f2; }
html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; direction: rtl; text-align: right; color: var(--ink); }
.stApp { background: radial-gradient(900px 400px at 15% -10%, #dceafe 0%, transparent 55%), radial-gradient(700px 350px at 90% 0%, #dcf5e5 0%, transparent 52%), linear-gradient(135deg, var(--bg-a) 0%, var(--bg-b) 100%); }
.block-container { max-width: 1160px; padding-top: 1.25rem; padding-bottom: 2rem; }
.hero { background: linear-gradient(120deg, #ffffff 0%, #f6fbff 100%); border: 1px solid #dce7f6; border-radius: 18px; padding: 1.1rem 1.1rem; box-shadow: 0 12px 30px rgba(15, 36, 56, 0.08); margin-bottom: .85rem; }
.hero h1 { margin: 0; font-size: 1.85rem; font-weight: 800; color: #0a2c55; }
.hero p { margin: .4rem 0 0 0; color: var(--muted); font-size: 1rem; font-weight: 500; }
.stats-grid { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: .55rem; margin-bottom: .7rem; }
.stat { border: 1px solid #d8e4f4; border-radius: 14px; background: var(--card); padding: .65rem .75rem; }
.stat .k { color: var(--muted); font-size: .82rem; margin-bottom: .2rem; }
.stat .v { color: #0a2c55; font-size: 1.08rem; font-weight: 800; }
.panel { border: 1px solid var(--line); border-radius: 16px; background: var(--card); padding: .9rem .9rem .6rem .9rem; box-shadow: 0 7px 22px rgba(15, 36, 56, 0.06); }
.panel-title { margin: 0 0 .4rem 0; color: #163a63; font-weight: 800; font-size: 1.05rem; }
.tip { border: 1px dashed #bfd6f3; border-radius: 12px; background: #f8fbff; color: #2a4a71; padding: .6rem .7rem; margin-bottom: .55rem; font-size: .95rem; }
.stButton button { border: 0; border-radius: 12px; width: 100%; background: linear-gradient(90deg, var(--primary) 0%, var(--primary-2) 100%); color: white; font-weight: 800; font-size: 1.01rem; padding: .62rem .9rem; }
.acc-row { display: grid; grid-template-columns: repeat(3, minmax(110px, 1fr)); gap: .5rem; margin-top: .55rem; }
.acc { border: 1px solid #d8e4f4; border-radius: 12px; background: #fbfdff; padding: .55rem .65rem; }
.acc .t { font-size: .79rem; color: var(--muted); }
.acc .n { font-size: 1.15rem; font-weight: 800; color: #10335c; }
@media (max-width: 920px) { .stats-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); } .acc-row { grid-template-columns: 1fr; } }
</style>
"""

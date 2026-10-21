"""Tampilan Streamlit untuk RGB_revisi.ipynb
Pengolahan citra lesi kulit (DermaMNIST 224 x 224).

Jalankan:  python -m streamlit run app_revisi.py
"""
import io
import math
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import streamlit as st
from matplotlib.patches import Circle, ConnectionPatch, Rectangle
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from skimage import color, draw, exposure, filters, measure, morphology

st.set_page_config(page_title="DermaMNIST · Pengolahan Citra Lesi Kulit", page_icon="🔬", layout="wide")
st.markdown(
    """
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg, #0f2b46 0%, #1b6ca8 55%, #19a7a0 100%); color: #fff;
       padding: 1.5rem 1.9rem; border-radius: 18px; margin-bottom: 1.2rem;}
.hero h1 {margin: 0; padding: 0; font-size: 2rem; color: #fff;}
.hero p {margin: .35rem 0 .2rem; opacity: .92;}
.chip {display: inline-block; background: rgba(255,255,255,.18); border-radius: 999px;
       padding: .15rem .85rem; margin: .5rem .45rem 0 0; font-size: .82rem;}
.sec {border-left: 6px solid #19a7a0; padding: .1rem 0 .1rem .9rem; margin: .3rem 0 1rem;}
.sec h2 {margin: 0; padding: 0; font-size: 1.55rem;}
.sec p {margin: .25rem 0 0; opacity: .75;}
div[data-testid="stMetric"] {background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.25);
       padding: .75rem 1rem; border-radius: 14px;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {padding: .4rem .7rem; border-radius: 10px;}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {background: rgba(25,167,160,.14);}
</style>
""",
    unsafe_allow_html=True,
)

DEFAULT_PATH = str(Path.home() / ".medmnist" / "dermamnist_224.npz")
FALLBACK_LABELS = {
    "0": "actinic keratoses and intraepithelial carcinoma", "1": "basal cell carcinoma",
    "2": "benign keratosis-like lesions", "3": "dermatofibroma", "4": "melanoma",
    "5": "melanocytic nevi", "6": "vascular lesions",
}
CHANNEL = [("Red", "red"), ("Green", "green"), ("Blue", "blue")]
PROPERTI = ("area", "perimeter", "equivalent_diameter_area", "feret_diameter_max",
            "axis_major_length", "axis_minor_length")
NAMA_KOLOM = {"area": "Luas (px²)", "perimeter": "Keliling (px)", "equivalent_diameter_area": "Diameter ekuivalen (px)",
              "feret_diameter_max": "Diameter Feret maks (px)", "axis_major_length": "Sumbu mayor (px)",
              "axis_minor_length": "Sumbu minor (px)"}
PROPERTI_LABEL = ("label", "centroid", "orientation", "axis_major_length", "axis_minor_length", "area",
                  "perimeter", "equivalent_diameter_area", "feret_diameter_max", "bbox")


# ============================================================================
# Utilitas tampilan & data
# ============================================================================
def header(judul, deskripsi=""):
    st.markdown(f'<div class="sec"><h2>{judul}</h2><p>{deskripsi}</p></div>', unsafe_allow_html=True)


def show(fig):
    with st.container(border=True):
        st.pyplot(fig)
    plt.close(fig)


def W(teks, n=18):
    return textwrap.fill(teks, n)


@st.cache_data
def get_labels():
    try:
        from medmnist import INFO
        return INFO["dermamnist"]["label"]
    except Exception:
        return FALLBACK_LABELS


@st.cache_data(show_spinner="Memuat dataset DermaMNIST...")
def load_sampel(path):
    """1 citra pertama dari setiap kelas pada train set (cell 1 notebook)."""
    data = np.load(path)
    images, labels = data["train_images"], data["train_labels"]
    n_kelas = len(get_labels())
    sampel = {}
    for img, lbl in zip(images, labels):
        lbl = int(np.squeeze(lbl))
        if lbl not in sampel:
            sampel[lbl] = img
        if len(sampel) == n_kelas:
            break
    return dict(sorted(sampel.items()))


# ============================================================================
# Fungsi pengolahan citra (diambil dari notebook)
# ============================================================================
def ke_gray(img):
    return (color.rgb2gray(img) * 255).astype(np.uint8)


def equalize_channel(channel):
    hist = ndi.histogram(channel, min=0, max=255, bins=256)
    cdf = hist.cumsum()
    return ((cdf / cdf.max())[channel] * 255).astype(np.uint8)


def equalize_rgb(img):
    return np.stack([equalize_channel(img[:, :, i]) for i in range(3)], axis=-1)


def plot_split_channel(img, judul, label_hist="Histogram"):
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.4], hspace=0.3, wspace=0.3)
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_rgb.imshow(img)
    ax_rgb.set_title(judul, fontsize=13, fontweight="bold")
    ax_rgb.axis("off")
    awal_panah = [(0.0, 0.30), (0.5, 0.0), (1.0, 0.30)]
    lengkung = [0.35, 0.0, -0.35]
    posisi_teks = [(-0.30, 0.40, "right"), (0.44, -0.10, "right"), (1.30, 0.40, "left")]
    for i, (nama, warna) in enumerate(CHANNEL):
        channel = img[:, :, i]
        ax_ch = fig.add_subplot(gs[1, i])
        ax_ch.imshow(channel, cmap="gray", vmin=0, vmax=255)
        ax_ch.axis("off")
        hist = ndi.histogram(channel, min=0, max=255, bins=256)
        ax_h = fig.add_subplot(gs[2, i])
        ax_h.bar(np.arange(256), hist, width=1.0, color=warna, alpha=0.45)
        ax_h.set_xlim(0, 255)
        ax_h.set_title(f"{label_hist} {nama}", fontsize=10)
        ax_h.set_xlabel("Gray Levels")
        ax_h.set_ylabel("Number of Pixels")
        fig.add_artist(ConnectionPatch(xyA=awal_panah[i], coordsA=ax_rgb.transAxes, xyB=(0.5, 1.03),
                                       coordsB=ax_ch.transAxes, arrowstyle="-|>", mutation_scale=28, lw=7,
                                       color=warna, connectionstyle=f"arc3,rad={lengkung[i]}"))
        tx, ty, ha = posisi_teks[i]
        ax_rgb.text(tx, ty, nama, transform=ax_rgb.transAxes, ha=ha, va="center", color=warna,
                    fontsize=15, fontweight="bold")
    return fig


def clahe_rgb(img, clip):
    f = img.astype(np.float64) / 255.0
    return np.stack([exposure.equalize_adapthist(f[:, :, i], clip_limit=clip) for i in range(3)], axis=-1)


def MeanFilter(image, filter_size):
    output = np.zeros(image.shape, np.uint8)
    result = 0
    r = 1 if filter_size == 9 else 2
    for j in range(r, image.shape[0] - r):
        for i in range(r, image.shape[1] - r):
            for y in range(-r, r + 1):
                for x in range(-r, r + 1):
                    result = result + image[j + y, i + x]
            output[j][i] = int(result / filter_size)
            result = 0
    return output


# ---- noise, filter, dan parameter evaluasi ----
def noise_salt_pepper(gray, jumlah=0.05, seed=0):
    rng = np.random.default_rng(seed)
    hasil, acak = gray.copy(), rng.random(gray.shape)
    hasil[acak < jumlah / 2] = 0
    hasil[acak > 1 - jumlah / 2] = 255
    return hasil


def noise_gaussian(gray, sigma=20, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(gray + rng.normal(0, sigma, gray.shape), 0, 255).astype(np.uint8)


def noise_speckle(gray, var=0.04, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(gray * (1 + rng.normal(0, np.sqrt(var), gray.shape)), 0, 255).astype(np.uint8)


def filter_3x3(citra):
    return {"Mean 3x3": ndi.convolve(citra.astype(float), np.full((3, 3), 1 / 9)),
            "Gaussian σ=1": ndi.gaussian_filter(citra.astype(float), sigma=1),
            "Median 3x3": ndi.median_filter(citra, size=3),
            "Min 3x3": ndi.minimum_filter(citra, size=3),
            "Max 3x3": ndi.maximum_filter(citra, size=3)}


def filter_7x7(gray):
    return {"Asli (grayscale)": gray,
            "Mean 7x7": ndi.convolve(gray.astype(float), np.full((7, 7), 1 / 49)),
            "Gaussian σ=2": ndi.gaussian_filter(gray.astype(float), sigma=2),
            "Median 7x7": filters.median(gray, np.ones((7, 7)))}


def hitung_mse(f, g):
    return np.mean((f.astype(float) - g.astype(float)) ** 2)


def hitung_psnr(f, g):
    mse = hitung_mse(f, g)
    return np.inf if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def hitung_enl(g, patch):
    p = g[patch].astype(float)
    return np.inf if p.var() == 0 else p.mean() ** 2 / p.var()


def hitung_cnr(g, mask_lesi):
    a, b = g[mask_lesi].astype(float), g[~mask_lesi].astype(float)
    return abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var())


def hitung_nm(f, g):
    return g.mean() / f.mean()


def evaluasi(f, g, patch, mask_lesi):
    return {"MSE": hitung_mse(f, g), "PSNR (dB)": hitung_psnr(f, g), "ENL": hitung_enl(g, patch),
            "CNR": hitung_cnr(g, mask_lesi), "NM": hitung_nm(f, g)}


def cari_patch_seragam(gray, ukuran=25):
    g = gray.astype(float)
    rata, rata2 = ndi.uniform_filter(g, ukuran), ndi.uniform_filter(g ** 2, ukuran)
    std = np.sqrt(np.maximum(rata2 - rata ** 2, 0))
    h = ukuran // 2
    std[:h, :] = np.inf
    std[-h:, :] = np.inf
    std[:, :h] = np.inf
    std[:, -h:] = np.inf
    r, c = np.unravel_index(np.argmin(std), std.shape)
    return slice(r - h, r + h + 1), slice(c - h, c + h + 1)


def daerah_lesi_kasar(gray):
    med = filters.median(gray, np.ones((7, 7)))
    return med <= filters.threshold_otsu(med)


# ---- hair removal & segmentasi ----
def se_garis(panjang, sudut):
    se = np.zeros((panjang, panjang), dtype=bool)
    c = panjang // 2
    dx, dy = np.cos(np.deg2rad(sudut)) * c, np.sin(np.deg2rad(sudut)) * c
    rr, cc = draw.line(int(round(c - dy)), int(round(c - dx)), int(round(c + dy)), int(round(c + dx)))
    se[rr, cc] = True
    return se


def hapus_rambut(img, ambang=14, panjang_garis=15, min_luas=40):
    kanal_v = img.max(axis=2).astype(float)
    blackhat = morphology.black_tophat(kanal_v, morphology.disk(5))
    kandidat = ndi.binary_dilation(blackhat > ambang, structure=morphology.disk(1))
    garis = np.zeros_like(kandidat)
    for sudut in range(0, 180, 15):
        garis |= ndi.binary_opening(kandidat, structure=se_garis(panjang_garis, sudut))
    garis = morphology.remove_small_objects(garis, max_size=min_luas)
    mask_rambut = ndi.binary_dilation(garis, iterations=5, mask=kandidat)
    mask_rambut = ndi.binary_dilation(mask_rambut, structure=morphology.disk(1))
    closing_rgb = np.stack([morphology.closing(img[:, :, i], morphology.disk(5)) for i in range(3)], axis=-1)
    img_bersih = np.where(mask_rambut[:, :, None], closing_rgb, img).astype(np.uint8)
    return img_bersih, mask_rambut, blackhat


def skor_objek(label_img):
    """skor = luas x (1 - jarak rata-rata piksel ke pusat)^2 x (1 - porsi tepi citra yang disentuh)^4"""
    h, w = label_img.shape
    yy, xx = np.mgrid[:h, :w]
    jarak = np.hypot(yy - h / 2, xx - w / 2) / np.hypot(h / 2, w / 2)
    tepi = np.zeros(label_img.shape, dtype=bool)
    tepi[[0, -1], :] = True
    tepi[:, [0, -1]] = True
    skor = {}
    for p in measure.regionprops(label_img):
        objek = label_img == p.label
        skor[p.label] = p.area * (1 - jarak[objek].mean()) ** 2 * (1 - (objek & tepi).sum() / tepi.sum()) ** 4
    return skor


def pilih_objek_lesi(mask):
    label_img = measure.label(mask)
    skor = skor_objek(label_img)
    if not skor:
        return mask
    return label_img == max(skor, key=skor.get)


@st.cache_data(show_spinner=False)
def rambut(kelas, ambang=14):
    return hapus_rambut(SAMPEL[kelas], ambang=ambang)


@st.cache_data(show_spinner="Menjalankan segmentasi...")
def segmentasi_lesi(kelas):
    img_bersih, mask_rambut, _ = rambut(kelas)
    gray = ke_gray(img_bersih)
    median = filters.median(gray, np.ones((7, 7)))
    thr = filters.threshold_otsu(median)
    biner = median <= thr
    bersih = morphology.remove_small_objects(biner, max_size=300)
    bersih = morphology.opening(bersih, morphology.disk(3))
    bersih = morphology.closing(bersih, morphology.disk(5))
    bersih = ndi.binary_fill_holes(bersih)
    kandidat = bersih
    lesi = pilih_objek_lesi(kandidat)
    if not lesi.any():
        kandidat = ndi.binary_fill_holes(biner)
        lesi = pilih_objek_lesi(kandidat)
    return {"img": SAMPEL[kelas], "img_bersih": img_bersih, "mask_rambut": mask_rambut, "gray": gray,
            "median": median, "thr": thr, "biner": biner, "bersih": bersih, "kandidat": kandidat, "lesi": lesi}


@st.cache_data(show_spinner="Memberi label pada objek...")
def labeling_objek(kelas):
    """measure.label pada objek hasil morphological filtering + tabel regionprops_table (seperti materi)."""
    s = segmentasi_lesi(kelas)
    label_img = measure.label(s["kandidat"])
    tabel = pd.DataFrame(measure.regionprops_table(label_img, properties=PROPERTI_LABEL))
    tabel = tabel.rename(columns={"axis_major_length": "major_axis_length", "axis_minor_length": "minor_axis_length"})
    label_lesi = int(label_img[s["lesi"]].max()) if s["lesi"].any() else 0
    tabel["skor"] = tabel["label"].map(skor_objek(label_img))
    tabel["lesi_terpilih"] = tabel["label"] == label_lesi
    tabel.insert(0, "kelas", kelas)
    tabel.insert(1, "nama_kelas", NAMA[str(kelas)])
    return label_img, tabel


def ukur_lesi(kelas):
    s = segmentasi_lesi(kelas)
    label_img = measure.label(s["lesi"])
    props = measure.regionprops(label_img)
    return s, label_img, (props[0] if props else None)


def titik_feret(mask):
    kontur = np.vstack(measure.find_contours(mask.astype(float), 0.5))
    hull = kontur[ConvexHull(kontur).vertices]
    jarak = squareform(pdist(hull))
    i, j = np.unravel_index(np.argmax(jarak), jarak.shape)
    return hull[i], hull[j]


@st.cache_data(show_spinner="Menguji filter terhadap noise...")
def uji_noise(kelas, sp=0.05, sg=20, sv=0.04):
    gray = ke_gray(SAMPEL[kelas])
    patch, lesi = cari_patch_seragam(gray), daerah_lesi_kasar(gray)
    noisy = {"Salt & pepper": noise_salt_pepper(gray, sp), "Gaussian": noise_gaussian(gray, sg),
             "Speckle": noise_speckle(gray, sv)}
    hasil, baris = {}, []
    for nama_noise, citra_noise in noisy.items():
        hasil[nama_noise] = filter_3x3(citra_noise)
        baris.append({"Noise": nama_noise, "Filter": "Tanpa filter", **evaluasi(gray, citra_noise, patch, lesi)})
        for nama_filter, g in hasil[nama_noise].items():
            baris.append({"Noise": nama_noise, "Filter": nama_filter, **evaluasi(gray, g, patch, lesi)})
    return noisy, hasil, pd.DataFrame(baris)


@st.cache_data(show_spinner=False)
def clahe_semua(clip):
    return {k: clahe_rgb(i, clip) for k, i in SAMPEL.items()}


@st.cache_data(show_spinner="Menghitung mean filter from scratch...")
def mean_scratch(kelas):
    g = ke_gray(SAMPEL[kelas]).astype(int)
    return g, MeanFilter(g, 9), ndi.convolve(g.astype(float), np.full((3, 3), 1 / 9))


# ============================================================================
# Halaman-halaman (sesuai bagian pada notebook)
# ============================================================================
def sec_data():
    header("1 · Data & Contoh Citra", "Satu citra pertama dari setiap kelas penyakit pada train set DermaMNIST.")
    contoh = next(iter(SAMPEL.values()))
    m = st.columns(4)
    m[0].metric("Jumlah kelas", len(SAMPEL))
    m[1].metric("Ukuran citra", f"{contoh.shape[0]} × {contoh.shape[1]}")
    m[2].metric("Channel warna", contoh.shape[2])
    m[3].metric("Tipe data", str(contoh.dtype))
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.ravel()
    for ax, (k, img) in zip(axes, SAMPEL.items()):
        ax.imshow(img)
        ax.set_title(f"Kelas {k}\n{NAMA[str(k)]}", fontsize=11)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Contoh Citra Setiap Kelas Penyakit (DermaMNIST)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5)
    show(fig)
    st.dataframe(pd.DataFrame([{"Kelas": k, "Nama kelas": NAMA[str(k)], "dtype": str(i.dtype), "shape": str(i.shape),
                                "min": int(i.min()), "max": int(i.max())} for k, i in SAMPEL.items()]).set_index("Kelas"))


def sec_hist():
    header("2 · Histogram RGB & Equalization",
           "Tiap channel dipisah → citra grayscale + histogram, lalu diequalisasi memakai cumulative histogram (CDF).")
    st.latex(r"I_{eq}(u,v)=\frac{CDF\left(I(u,v)\right)}{CDF_{max}}\times 255")
    img, nama = SAMPEL[kelas], f"Kelas {kelas}: {NAMA[str(kelas)]}"
    eq = equalize_rgb(img)
    t1, t2, t3 = st.tabs(["🖼️ Citra asli", "✨ Equalized RGB", "📋 Brightness & contrast"])
    with t1:
        show(plot_split_channel(img, f"Citra Asli\n{nama}", "Histogram"))
    with t2:
        show(plot_split_channel(eq, f"Equalized RGB\n{nama}", "Histogram Equalized"))
    with t3:
        st.caption("Brightness = rata-rata intensitas · Contrast = standar deviasi intensitas.")
        st.dataframe(pd.DataFrame({
            "Channel": [n for n, _ in CHANNEL],
            "Brightness asli": [img[:, :, i].mean() for i in range(3)],
            "Brightness equalized": [eq[:, :, i].mean() for i in range(3)],
            "Contrast asli (std)": [img[:, :, i].std() for i in range(3)],
            "Contrast equalized (std)": [eq[:, :, i].std() for i in range(3)],
            "Rentang asli": [f"{img[:, :, i].min()} - {img[:, :, i].max()}" for i in range(3)],
            "Rentang equalized": [f"{eq[:, :, i].min()} - {eq[:, :, i].max()}" for i in range(3)],
        }).round(2).set_index("Channel"))


def sec_clahe():
    header("2b · CLAHE (Adaptive Histogram Equalization)",
           "Histogram dihitung per area kecil sehingga kontras lokal meningkat, berbeda dari equalization global.")
    clip = st.slider("Clip limit", 0.005, 0.100, 0.030, 0.005, format="%.3f")
    hasil = clahe_semua(clip)
    fig, axes = plt.subplots(2, len(hasil), figsize=(3.2 * len(hasil), 7))
    for j, (k, c) in enumerate(hasil.items()):
        axes[0, j].imshow(SAMPEL[k])
        axes[0, j].set_title(f"Asli - Kelas {k}\n{W(NAMA[str(k)], 16)}", fontsize=9)
        axes[1, j].imshow(c)
        axes[1, j].set_title(f"CLAHE - Kelas {k}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    show(fig)


def sec_teori():
    header("3 · Filtering: Teori & Contoh Perhitungan",
           "Filter menggabungkan nilai sebuah piksel dengan piksel-piksel tetangganya.")
    st.markdown("""
| Jenis | Filter | Cara kerja | Fungsi Python |
|---|---|---|---|
| Linear | **Mean** | rata-rata piksel + tetangga, bobot 1/9 (3×3) | `ndi.convolve` |
| Linear | **Gaussian** | rata-rata berbobot, bobot terbesar di pusat kernel | `ndi.gaussian_filter` |
| Non-linear | **Median** | diganti median tetangga; baik untuk *salt & pepper*, tepi tetap tajam | `ndi.median_filter` |
| Non-linear | **Min** | nilai terkecil tetangga, menghilangkan *salt noise* | `ndi.minimum_filter` |
| Non-linear | **Max** | nilai terbesar tetangga, menghilangkan *pepper noise* | `ndi.maximum_filter` |
""")
    st.markdown("##### 📐 Parameter evaluasi (*f* = citra asli, *g* = hasil filter)")
    rumus = [("MSE", r"\frac{1}{MN}\sum\sum (f-g)^2", "makin kecil, makin mirip asli"),
             ("PSNR", r"10\log_{10}\frac{255^2}{MSE}", "makin besar, makin baik"),
             ("ENL", r"\frac{M^2}{SD^2}", "makin besar, noise makin lemah"),
             ("CNR", r"\frac{|\mu_1-\mu_2|}{\sqrt{\sigma_1^2+\sigma_2^2}}", "lesi vs kulit; makin besar makin mudah dipisah"),
             ("NM", r"\frac{M_{filtered}}{M_{original}}", "mendekati 1 = informasi terjaga")]
    for col, (n, f, ket) in zip(st.columns(5), rumus):
        with col.container(border=True):
            st.markdown(f"**{n}**")
            st.latex(f)
            st.caption(ket)

    st.markdown("---")
    st.subheader("3.1 Contoh perhitungan Mean & Median filter")
    blok = np.array([[104, 100, 108], [99, 106, 98], [95, 90, 85]])
    contoh = np.array([[5, 6, 5, 7, 8], [9, 10, 1, 3, 2], [5, 6, 10, 3, 2], [7, 4, 4, 5, 1], [2, 3, 5, 3, 3]])
    med = contoh.copy()
    for j in range(1, contoh.shape[0] - 1):
        for i in range(1, contoh.shape[1] - 1):
            med[j, i] = np.median(contoh[j - 1:j + 2, i - 1:i + 2])
    a, b = st.columns(2)
    with a.container(border=True):
        st.markdown("**Mean filter 3×3** (kernel 1/9)")
        st.dataframe(pd.DataFrame(blok))
        st.metric("Hasil pada piksel pusat (106)", f"{(blok * np.full((3, 3), 1 / 9)).sum():.4f}")
    with b.container(border=True):
        st.markdown("**Median filter 3×3**")
        c1, c2 = st.columns(2)
        c1.caption("Input")
        c1.dataframe(pd.DataFrame(contoh))
        c2.caption("Hasil")
        c2.dataframe(pd.DataFrame(med))
        st.caption(f"F(2,2) = median {sorted(contoh[0:3, 0:3].ravel().tolist())} = {med[1, 1]}")

    st.markdown("##### Mean filter *from scratch* vs `ndi.convolve`")
    gray, scratch, nd = mean_scratch(kelas)
    selisih = np.abs(scratch[1:-1, 1:-1] - np.floor(nd[1:-1, 1:-1] + 1e-9))
    st.info(f"Kelas {kelas}: selisih maksimum antara kedua cara = **{selisih.max():.0f}**")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, tt in zip(axes, (gray, scratch, nd),
                          ("Grayscale asli", "Mean filter 3x3 (from scratch, tepi = 0)", "Mean filter 3x3 (ndi.convolve)")):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(tt)
        ax.axis("off")
    fig.tight_layout()
    show(fig)


def sec_noise():
    header("3.2 · Uji Filter terhadap Noise",
           "Citra grayscale diberi tiga jenis noise lalu difilter (Mean, Gaussian, Median, Min, Max; kernel 3×3).")
    with st.expander("⚙️ Parameter noise (nilai awal sama dengan notebook)"):
        c = st.columns(3)
        sp = c[0].slider("Salt & pepper (proporsi)", 0.01, 0.20, 0.05, 0.01)
        sg = c[1].slider("Gaussian (σ)", 5, 60, 20)
        sv = c[2].slider("Speckle (varians)", 0.01, 0.20, 0.04, 0.01)
    gray = ke_gray(SAMPEL[kelas])
    noisy, hasil, tabel = uji_noise(kelas, sp, sg, sv)

    st.markdown("##### 🏆 Filter dengan PSNR tertinggi")
    for col, (nn, t) in zip(st.columns(3), tabel[tabel["Filter"] != "Tanpa filter"].groupby("Noise", sort=False)):
        best = t.loc[t["PSNR (dB)"].idxmax()]
        col.metric(nn, best["Filter"])
        col.caption(f"PSNR = {best['PSNR (dB)']:.2f} dB")

    t1, t2, t3 = st.tabs(["🖼️ Citra ber-noise", "🧪 Hasil filter", "📋 Tabel metrik"])
    with t1:
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        axes[0].set_title("Grayscale asli")
        for ax, (nama, im) in zip(axes[1:], noisy.items()):
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{nama} noise\nPSNR = {hitung_psnr(gray, im):.2f} dB")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        show(fig)
    with t2:
        nama_filter = list(next(iter(hasil.values())).keys())
        fig, axes = plt.subplots(len(hasil), len(nama_filter), figsize=(18, 11))
        for i, (nn, hf) in enumerate(hasil.items()):
            for j, (nf, g) in enumerate(hf.items()):
                axes[i, j].imshow(g, cmap="gray", vmin=0, vmax=255)
                axes[i, j].set_title(f"{nn} → {nf}\nPSNR = {hitung_psnr(gray, g):.2f} dB", fontsize=10)
                axes[i, j].axis("off")
        fig.tight_layout()
        show(fig)
    with t3:
        st.dataframe(tabel.round(3).replace([np.inf, -np.inf], np.nan).set_index(["Noise", "Filter"]))

    st.markdown("---")
    if st.toggle("📊 Tampilkan rekap seluruh kelas (lebih lama)"):
        semua = []
        for k in SAMPEL:
            t = uji_noise(k, sp, sg, sv)[2].copy()
            t.insert(0, "Kelas", k)
            semua.append(t)
        tn = pd.concat(semua, ignore_index=True)
        urut_f = ["Tanpa filter", "Mean 3x3", "Gaussian σ=1", "Median 3x3", "Min 3x3", "Max 3x3"]
        urut_n = ["Salt & pepper", "Gaussian", "Speckle"]
        for col, metrik in zip(st.columns(2), ("PSNR (dB)", "ENL")):
            col.markdown(f"**Rata-rata {metrik} seluruh kelas**")
            piv = (tn.pivot_table(index="Filter", columns="Noise", values=metrik, aggfunc="mean")
                   .loc[urut_f, urut_n].replace([np.inf, -np.inf], np.nan))
            col.dataframe(piv.style.format("{:.2f}").background_gradient(cmap="YlGnBu", axis=None))
        rata = tn[tn["Filter"] != "Tanpa filter"].groupby(["Noise", "Filter"])["PSNR (dB)"].mean()
        for nn in urut_n:
            st.write(f"**{nn}**: filter dengan PSNR rata-rata tertinggi = `{rata[nn].idxmax()}`")


def sec_cmp():
    header("3.3 · Mean vs Median sebelum Segmentasi",
           "Denoising citra asli (jendela 7×7) agar nilai keabuan lesi dan kulit lebih seragam sebelum Otsu.")
    gray = ke_gray(SAMPEL[kelas])
    hasil, patch, lesi = filter_7x7(gray), cari_patch_seragam(gray), daerah_lesi_kasar(gray)
    fig, axes = plt.subplots(4, 4, figsize=(18, 17), gridspec_kw={"height_ratios": [1, 0.8, 0.55, 1]})
    baris = []
    for j, (nama, im) in enumerate(hasil.items()):
        axes[0, j].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[0, j].add_patch(Rectangle((patch[1].start, patch[0].start), patch[1].stop - patch[1].start,
                                       patch[0].stop - patch[0].start, fill=False, color="yellow", lw=2))
        axes[0, j].set_title(nama, fontsize=13)
        axes[0, j].axis("off")
        zoom = axes[1, j].imshow(im[patch], cmap="gray", interpolation="nearest")
        axes[1, j].set_title(f"Zoom daerah seragam, std = {im[patch].std():.2f}", fontsize=11)
        axes[1, j].axis("off")
        fig.colorbar(zoom, ax=axes[1, j], fraction=0.046)
        hist = ndi.histogram(im, min=0, max=255, bins=256)
        thr = filters.threshold_otsu(np.asarray(im))
        axes[2, j].plot(hist, color="tab:blue")
        axes[2, j].axvline(thr, color="red", ls="--", label=f"Otsu = {thr:.0f}")
        axes[2, j].set_xlim(0, 255)
        axes[2, j].set_xlabel("gray value")
        axes[2, j].set_ylabel("number of pixels")
        axes[2, j].legend(fontsize=9)
        biner = np.asarray(im) <= thr
        n_objek = measure.label(biner).max()
        axes[3, j].imshow(biner, cmap="gray")
        axes[3, j].set_title(f"Otsu biner: {n_objek} objek", fontsize=12)
        axes[3, j].axis("off")
        baris.append({"Filter": nama, "Std patch": im[patch].std(), "ENL": hitung_enl(im, patch),
                      "CNR": hitung_cnr(im, lesi), "NM": hitung_nm(gray, im), "Jumlah objek Otsu": n_objek})
    fig.tight_layout()
    show(fig)
    st.dataframe(pd.DataFrame(baris).set_index("Filter").round(3))
    st.success(
        "**Kesimpulan: filter yang dipakai untuk segmentasi adalah Median 7×7.**\n\n"
        "Median paling baik untuk *salt & pepper* (Mean/Gaussian hanya menyebarkan bintik), sedangkan untuk Gaussian "
        "dan speckle noise Mean/Gaussian sedikit lebih unggul. Pada citra asli, Median meredam variasi keabuan di "
        "daerah seragam dan **tepi lesi tetap tajam**, sehingga hasil Otsu lebih bersih."
    )


def sec_hair():
    header("4 · Hair Removal (Morphological Filtering)",
           "Rambut berupa garis gelap, tipis, dan panjang; dihilangkan sebelum segmentasi.")
    with st.expander("🧩 Langkah-langkah"):
        st.markdown(
            "1. **Kanal V = max(R, G, B)** → **black-hat** (closing − citra, SE *disk* r = 5) menonjolkan struktur gelap tipis.\n"
            "2. **Threshold** black-hat → kandidat rambut, **dilasi 1 piksel**.\n"
            "3. **Opening dengan SE garis** (15 piksel, 12 arah 0°–165°) → hanya struktur panjang & lurus yang bertahan.\n"
            "4. **Remove small objects** + dilasi (dibatasi di dalam kandidat) agar helai tertutup penuh.\n"
            "5. **Apply mask**: piksel rambut diganti hasil *closing* tiap channel RGB.")
    c1, c2 = st.columns([3, 1])
    ambang = c1.slider("Ambang black-hat (nilai awal notebook = 14)", 5, 40, 14)
    semua = c2.toggle("Semua kelas")
    ks = list(SAMPEL) if semua else [kelas]
    n = len(ks)
    fig, axes = plt.subplots(4 if semua else 1, n if semua else 4, squeeze=False,
                             figsize=(3.2 * n, 13.5) if semua else (16, 4.3))
    for j, k in enumerate(ks):
        bersih, mask, bh = rambut(k, ambang)
        panel = [(SAMPEL[k], None, f"Asli - Kelas {k}\n{W(NAMA[str(k)])}"), (bh, "gray", "Black-hat"),
                 (mask, "gray", f"Mask rambut ({mask.mean() * 100:.1f}% piksel)"), (bersih, None, "Setelah hair removal")]
        for p, (im, cm, tt) in enumerate(panel):
            ax = axes[p, j] if semua else axes[0, p]
            ax.imshow(im, cmap=cm)
            ax.set_title(tt, fontsize=10)
            ax.axis("off")
    fig.tight_layout()
    show(fig)


def sec_seg():
    header("5 · Segmentasi Lesi",
           "Hair removal → grayscale → median 7×7 → Otsu → morphological filtering → pemilihan objek lesi.")
    s = segmentasi_lesi(kelas)
    m = st.columns(4)
    m[0].metric("Threshold Otsu", int(s["thr"]))
    m[1].metric("Luas lesi", f"{int(s['lesi'].sum())} px²")
    m[2].metric("Lesi dari citra", f"{s['lesi'].mean() * 100:.1f}%")
    m[3].metric("Rambut terdeteksi", f"{s['mask_rambut'].mean() * 100:.1f}%")
    with st.expander("🎯 Cara memilih objek lesi"):
        st.latex(r"skor = luas \times (1-\bar{d})^2 \times (1-t)^4")
        st.caption("d̄ = jarak rata-rata piksel objek ke pusat citra (0 = pusat, 1 = sudut), t = porsi tepi citra yang "
                   "disentuh objek. Lesi biasanya di tengah, sedangkan bayangan di pinggir (vignetting) skornya kecil.")
    fig, axes = plt.subplots(2, 5, figsize=(22, 9.5))
    fig.suptitle(f"Segmentasi - Kelas {kelas}: {NAMA[str(kelas)]}", fontsize=15, fontweight="bold")
    axes[0, 0].imshow(s["img"]); axes[0, 0].set_title("1. Citra asli")
    axes[0, 1].imshow(s["img_bersih"]); axes[0, 1].set_title("2. Setelah hair removal")
    axes[0, 2].imshow(s["gray"], cmap="gray", vmin=0, vmax=255); axes[0, 2].set_title("3. Grayscale")
    axes[0, 3].imshow(s["median"], cmap="gray", vmin=0, vmax=255); axes[0, 3].set_title("4. Median filter 7x7")
    axes[0, 4].plot(ndi.histogram(s["median"], min=0, max=255, bins=256))
    axes[0, 4].axvline(s["thr"], color="red", ls="--", label=f"Otsu = {s['thr']}")
    axes[0, 4].set_xlim(0, 255)
    axes[0, 4].set_title("5. Histogram + threshold Otsu")
    axes[0, 4].set_xlabel("gray value")
    axes[0, 4].set_ylabel("number of pixels")
    axes[0, 4].legend()
    axes[1, 0].imshow(s["biner"], cmap="gray"); axes[1, 0].set_title(f"6. Biner (median ≤ {s['thr']})")
    axes[1, 1].imshow(s["bersih"], cmap="gray"); axes[1, 1].set_title("7. Morphological filtering")
    axes[1, 2].imshow(s["lesi"], cmap="gray"); axes[1, 2].set_title("8. Mask lesi")
    axes[1, 3].imshow(s["img_bersih"])
    axes[1, 3].contour(s["lesi"], [0.5], colors="lime", linewidths=2)
    axes[1, 3].set_title("9. Kontur lesi")
    axes[1, 4].imshow(np.where(s["lesi"][:, :, None], s["img_bersih"], 0))
    axes[1, 4].set_title("10. Hasil segmentasi (apply mask)")
    for ax in axes.ravel():
        if ax is not axes[0, 4]:
            ax.axis("off")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    show(fig)


def rekap_lesi():
    """Rekap luas & diameter lesi terpilih untuk semua kelas."""
    rekap = []
    for k in SAMPEL:
        sk, _, pk = ukur_lesi(k)
        if pk is None:
            continue
        baris = {"Kelas": k, "Nama kelas": NAMA[str(k)]}
        baris.update({NAMA_KOLOM[x]: getattr(pk, x) for x in PROPERTI})
        baris["Luas (% citra)"] = pk.area / sk["lesi"].size * 100
        rekap.append(baris)
    return pd.DataFrame(rekap).set_index("Kelas")


def ke_csv(df, index=False):
    """DataFrame -> bytes CSV (utf-8-sig agar karakter seperti ² terbaca benar di Excel)."""
    return df.to_csv(index=index).encode("utf-8-sig")


def ke_excel(sheets):
    """{nama_sheet: (DataFrame, pakai_index)} -> bytes .xlsx, atau None jika openpyxl belum terpasang."""
    buf = io.BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for nama, (df, idx) in sheets.items():
                df.to_excel(writer, sheet_name=nama, index=idx)
    except ImportError:
        return None
    return buf.getvalue()


def tombol_unduh(kolom, label, data, nama_file, mime):
    if data is None:
        kolom.caption("Excel butuh `openpyxl`: `pip install openpyxl`")
    else:
        kolom.download_button(label, data=data, file_name=nama_file, mime=mime)


MIME_CSV = "text/csv"
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def sec_ukur():
    header("6 · Labeling & Region Properties",
           "Setiap objek hasil segmentasi diberi label (`measure.label`), dihitung propertinya "
           "(`measure.regionprops_table`), lalu tabelnya di-export ke CSV/Excel. Satuan: piksel (citra 224 × 224).")
    t1, t2, t3 = st.tabs(["🏷️ 6.1 Labeling objek", "📏 6.2 Luas & diameter lesi", "💾 6.3 Rekap & export"])

    # ---------------- 6.1 Labeling objek ----------------
    with t1:
        s = segmentasi_lesi(kelas)
        label_img, tabel = labeling_objek(kelas)
        m = st.columns(3)
        m[0].metric("Jumlah objek berlabel", len(tabel))
        m[1].metric("Label lesi terpilih",
                    int(tabel.loc[tabel["lesi_terpilih"], "label"].iloc[0]) if tabel["lesi_terpilih"].any() else "-")
        m[2].metric("Luas lesi", f"{int(s['lesi'].sum())} px²")
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
        axes[0].imshow(s["kandidat"], cmap="gray")
        axes[0].set_title("Citra biner (setelah morphological filtering)")
        axes[1].imshow(color.label2rgb(label_img, bg_label=0))
        axes[1].set_title(f"Label image: {label_img.max()} objek")
        axes[2].imshow(s["img_bersih"])
        axes[2].contour(s["lesi"], [0.5], colors="lime", linewidths=2)
        axes[2].set_title("Objek pada citra (hijau = lesi terpilih)")
        for _, r in tabel.iterrows():
            for ax in axes[1:]:
                ax.text(r["centroid-1"], r["centroid-0"], str(int(r["label"])), color="white", fontsize=12,
                        fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6))
            axes[2].add_patch(Rectangle((r["bbox-1"], r["bbox-0"]), r["bbox-3"] - r["bbox-1"], r["bbox-2"] - r["bbox-0"],
                                        fill=False, color="lime" if r["lesi_terpilih"] else "deepskyblue", lw=2))
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        show(fig)
        st.markdown("##### Tabel region properties semua objek berlabel")
        st.dataframe(tabel.round(3), hide_index=True)
        c = st.columns(2)
        tombol_unduh(c[0], f"⬇️ CSV labeling kelas {kelas}", ke_csv(tabel), f"labeling_kelas_{kelas}.csv", MIME_CSV)
        tombol_unduh(c[1], f"⬇️ Excel labeling kelas {kelas}", ke_excel({f"Kelas {kelas}": (tabel, False)}),
                     f"labeling_kelas_{kelas}.xlsx", MIME_XLSX)

    # ---------------- 6.2 Luas & diameter lesi ----------------
    with t2:
        s, label_img, p = ukur_lesi(kelas)
        if p is None:
            st.warning("Tidak ada objek lesi yang terdeteksi pada kelas ini.")
        else:
            m = st.columns(5)
            m[0].metric("Luas", f"{p.area:.0f} px²")
            m[1].metric("Keliling", f"{p.perimeter:.1f} px")
            m[2].metric("Diameter ekuivalen", f"{p.equivalent_diameter_area:.1f} px")
            m[3].metric("Diameter Feret maks", f"{p.feret_diameter_max:.1f} px")
            m[4].metric("Sumbu mayor / minor", f"{p.axis_major_length:.0f} / {p.axis_minor_length:.0f}")
            y0, x0 = p.centroid
            ori = p.orientation
            fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
            ax = axes[0]
            ax.imshow(s["lesi"], cmap="gray")
            ax.plot((x0, x0 + math.cos(ori) * 0.5 * p.axis_minor_length),
                    (y0, y0 - math.sin(ori) * 0.5 * p.axis_minor_length), "-r", linewidth=2.5)
            ax.plot((x0, x0 - math.sin(ori) * 0.5 * p.axis_major_length),
                    (y0, y0 - math.cos(ori) * 0.5 * p.axis_major_length), "-r", linewidth=2.5)
            ax.plot(x0, y0, ".g", markersize=18)
            minr, minc, maxr, maxc = p.bbox
            ax.plot((minc, maxc, maxc, minc, minc), (minr, minr, maxr, maxr, minr), "-b", linewidth=2.5)
            ax.set_title("Region properties (bbox, centroid, sumbu mayor/minor)")
            ax = axes[1]
            ax.imshow(s["img_bersih"])
            ax.contour(s["lesi"], [0.5], colors="lime", linewidths=2)
            a, b = titik_feret(s["lesi"])
            ax.plot([a[1], b[1]], [a[0], b[0]], "-", color="yellow", linewidth=3,
                    label=f"Diameter Feret maks = {p.feret_diameter_max:.1f} px")
            ax.add_patch(Circle((x0, y0), p.equivalent_diameter_area / 2, fill=False, color="cyan", ls="--", lw=2,
                                label=f"Diameter ekuivalen = {p.equivalent_diameter_area:.1f} px"))
            ax.plot(x0, y0, ".g", markersize=14)
            ax.legend(loc="lower left", fontsize=10, framealpha=0.85)
            ax.set_title(f"Luas = {p.area:.0f} px² ({p.area / s['lesi'].size * 100:.1f}% citra)")
            for ax in axes:
                ax.set_xlim(0, s["lesi"].shape[1])
                ax.set_ylim(s["lesi"].shape[0], 0)
            fig.tight_layout()
            show(fig)
            st.caption("Bounding box (biru) · centroid (hijau) · sumbu mayor & minor (merah) · diameter Feret (kuning) · "
                       "lingkaran diameter ekuivalen (cyan)")
            st.dataframe(pd.DataFrame(measure.regionprops_table(label_img, properties=PROPERTI))
                         .rename(columns=NAMA_KOLOM).round(2), hide_index=True)

    # ---------------- 6.3 Rekap & export ----------------
    with t3:
        rekap = rekap_lesi()
        semua_objek = pd.concat([labeling_objek(k)[1] for k in SAMPEL], ignore_index=True)
        st.markdown("##### Rekap luas & diameter lesi semua kelas")
        st.dataframe(rekap.round(2))
        fig, axes = plt.subplots(1, len(SAMPEL), figsize=(3.2 * len(SAMPEL), 4.6), squeeze=False)
        for ax, k in zip(axes[0], SAMPEL):
            sk, _, pk = ukur_lesi(k)
            ax.imshow(sk["img_bersih"])
            ax.contour(sk["lesi"], [0.5], colors="lime", linewidths=1.5)
            judul = (f"Kelas {k}\nLuas = {pk.area:.0f} px²\nd_ekv = {pk.equivalent_diameter_area:.1f} | "
                     f"d_Feret = {pk.feret_diameter_max:.1f}") if pk is not None else f"Kelas {k}\n(tidak terdeteksi)"
            ax.set_title(judul, fontsize=9)
            ax.axis("off")
        fig.tight_layout()
        show(fig)

        st.markdown("##### Tabel labeling semua objek (semua kelas)")
        st.dataframe(semua_objek.round(3), hide_index=True)

        st.markdown("##### 💾 Export file")
        sheets = {"Semua objek": (semua_objek, False), "Rekap lesi": (rekap, True)}
        sheets.update({f"Kelas {k}": (t, False) for k, t in semua_objek.groupby("kelas")})
        c = st.columns(3)
        tombol_unduh(c[0], "⬇️ labeling_semua_objek.csv", ke_csv(semua_objek), "labeling_semua_objek.csv", MIME_CSV)
        tombol_unduh(c[1], "⬇️ rekap_lesi.csv", ke_csv(rekap, index=True), "rekap_lesi.csv", MIME_CSV)
        tombol_unduh(c[2], "⬇️ hasil_labeling.xlsx", ke_excel(sheets), "hasil_labeling.xlsx", MIME_XLSX)
        st.caption("File Excel berisi sheet *Semua objek*, *Rekap lesi*, dan satu sheet per kelas. CSV memakai pemisah "
                   "koma; jika di Excel semua kolom terbaca dalam satu kolom, buka file `.xlsx` atau gunakan "
                   "*Data → From Text/CSV*.")


PAGES = {
    "Data & Contoh Citra": sec_data,
    "Histogram & Equalization": sec_hist,
    "CLAHE": sec_clahe,
    "Filtering: Teori & Contoh": sec_teori,
    "Uji Filter terhadap Noise": sec_noise,
    "Mean vs Median": sec_cmp,
    "Hair Removal": sec_hair,
    "Segmentasi Lesi": sec_seg,
    "Labeling & Region Properties": sec_ukur,
}

# ============================================================================
# Sidebar, pemuatan data, dan routing halaman
# ============================================================================
with st.sidebar:
    st.markdown("## 🔬 DermaMNIST")
    st.caption("Pengolahan Citra Lesi Kulit · RGB_revisi.ipynb")
    menu = st.radio("Navigasi", list(PAGES), label_visibility="collapsed")
    st.divider()
    slot = st.container()
    with st.expander("📁 Dataset"):
        path = st.text_input("Lokasi dermamnist_224.npz", DEFAULT_PATH)

if not Path(path).is_file():
    st.error(f"File dataset tidak ditemukan: `{path}`")
    st.write("Perbaiki lokasi file di sidebar, atau unduh otomatis ke folder `~/.medmnist` dengan tombol berikut.")
    if st.button("⬇️ Unduh DermaMNIST 224 (medmnist)"):
        try:
            from medmnist import DermaMNIST
            with st.spinner("Mengunduh dataset..."):
                DermaMNIST(split="train", download=True, size=224)
            st.rerun()
        except ImportError:
            st.warning("Library `medmnist` belum terpasang: `pip install medmnist`.")
    st.stop()

if st.session_state.setdefault("_path", path) != path:  # dataset diganti -> buang cache lama
    st.cache_data.clear()
    st.session_state["_path"] = path

SAMPEL, NAMA = load_sampel(path), get_labels()
with slot:
    kelas = st.selectbox("Pilih kelas", list(SAMPEL), format_func=lambda k: f"Kelas {k}: {NAMA[str(k)]}")

st.markdown(
    f"""<div class="hero"><h1>🔬 Pengolahan Citra Lesi Kulit</h1>
<p>DermaMNIST (224 × 224 RGB): histogram, equalization, filtering, hair removal, segmentasi, labeling, dan pengukuran lesi.</p>
<span class="chip">Kelas {kelas}: {NAMA[str(kelas)]}</span><span class="chip">{menu}</span></div>""",
    unsafe_allow_html=True,
)
PAGES[menu]()

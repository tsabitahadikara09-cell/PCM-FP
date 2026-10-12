import os
import sys

import cv2
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PyQt5 import uic
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QLabel,
    QMessageBox,
    QTableWidgetItem,
)

# ----------------------------------------------------------------------------
# KONFIGURASI
# ----------------------------------------------------------------------------
UI_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.ui")

# Ukuran gambar MedMNIST: 28, 64, 128, atau 224.
# 28 terlalu kecil untuk segmentasi; 128 kompromi yang bagus (download lebih ringan dari 224).
IMG_SIZE = 128
SPLIT = "test"  # "train", "val", atau "test"

# Label DermaMNIST (sesuai INFO['dermamnist']['label'])
DERMA_LABELS = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}


# ----------------------------------------------------------------------------
# FUNGSI PENGOLAHAN CITRA (semua input/output: numpy RGB uint8)
# ----------------------------------------------------------------------------
def hist_equalization(rgb):
    """Histogram Equalization pada channel luminance (Y) supaya warna tidak bergeser."""
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def clahe_enhance(rgb, clip=2.0, grid=8):
    """AHE dengan CLAHE pada channel L (LAB)."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def contrast_stretching(rgb, low=2, high=98):
    """Contrast stretching per channel, memakai persentil supaya tahan outlier."""
    out = np.empty_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], (low, high))
        if hi - lo < 1:
            out[..., c] = rgb[..., c]
            continue
        ch = (rgb[..., c].astype(np.float32) - lo) * 255.0 / (hi - lo)
        out[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
    return out


def add_salt_pepper(rgb, density):
    """Tambah noise salt & pepper. density = proporsi piksel yang kena noise."""
    rng = np.random.default_rng()
    out = rgb.copy()
    r = rng.random(rgb.shape[:2])
    out[r < density / 2] = 0  # pepper
    out[(r >= density / 2) & (r < density)] = 255  # salt
    return out


def mse_psnr(ref, img):
    diff = ref.astype(np.float64) - img.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    psnr = float("inf") if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse)
    return mse, psnr


def _clean_mask(mask):
    """Ambil komponen terbesar & isi lubang -> mask (0/255) + contours-nya."""
    # kalau "lesi" menutupi mayoritas gambar, kemungkinan hasil threshold terbalik
    if mask.mean() / 255.0 > 0.6:
        mask = cv2.bitwise_not(mask)

    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k7)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final = np.zeros_like(mask)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(final, [biggest], -1, 255, thickness=cv2.FILLED)
        contours = [biggest]
    return final, contours


def segment_lesion(rgb, method):
    """Segmentasi lesi. Return (mask 0/255, overlay RGB)."""
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    if method.startswith("Otsu"):
        _, raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    elif method.startswith("K-Means"):
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(lab, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        lesion_cluster = int(np.argmin(centers[:, 0]))  # cluster paling gelap (L terendah)
        raw = (labels.reshape(h, w) == lesion_cluster).astype(np.uint8) * 255

    else:  # Watershed
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=2)
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1  # background = 1, objek >= 2
        markers[unknown == 255] = 0  # area yang akan diputuskan watershed
        markers = cv2.watershed(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), markers)
        raw = (markers > 1).astype(np.uint8) * 255

    mask, contours = _clean_mask(raw)

    colored = rgb.copy()
    colored[mask > 0] = (255, 0, 0)
    overlay = cv2.addWeighted(rgb, 0.65, colored, 0.35, 0)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    return mask, overlay


def render_histogram(original, enhanced=None):
    """Bikin gambar histogram RGB (numpy RGB). Garis putus = original, solid = enhanced."""
    fig = Figure(figsize=(3.2, 3.2), dpi=100, facecolor="#1e1e1e")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, facecolor="#1e1e1e")

    colors = ("#ff5555", "#55ff55", "#5599ff")
    for c, col in enumerate(colors):
        h0 = cv2.calcHist([original], [c], None, [256], [0, 256]).flatten()
        if enhanced is None:
            ax.plot(h0, color=col, linewidth=1.2)
        else:
            ax.plot(h0, color=col, linewidth=0.9, linestyle="--", alpha=0.6)
            h1 = cv2.calcHist([enhanced], [c], None, [256], [0, 256]).flatten()
            ax.plot(h1, color=col, linewidth=1.2)

    ax.set_xlim(0, 255)
    ax.tick_params(colors="#cccccc", labelsize=7)
    for s in ax.spines.values():
        s.set_color("#555555")
    ax.set_title(
        "RGB Histogram" if enhanced is None else "RGB Histogram (-- asli, — enhanced)",
        color="#dddddd",
        fontsize=8,
    )
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    return np.ascontiguousarray(buf)


def to_qimage(arr):
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


# ----------------------------------------------------------------------------
# DIALOG UTAMA
# ----------------------------------------------------------------------------
class MainDialog(QDialog):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)

        # state
        self.dataset = None
        self.class_indices = np.array([], dtype=int)
        self.ptr = -1
        self.current_rgb = None  # gambar dasar (dari dataset / upload)
        self.enhanced = None
        self.noisy = None
        self.filtered = None
        self.filter_info = None  # (nama, kernel)

        # label gambar: kita scale sendiri supaya aspect ratio terjaga
        self._shown = {}
        self._placeholder = {}
        for lbl in self.findChildren(QLabel):
            if lbl.objectName().startswith("lbl_"):
                lbl.setScaledContents(False)
                self._placeholder[lbl] = lbl.text()

        self.lineEditPsnrResult.setText("-")

        # signal
        self.comboJenisPenyakit.currentIndexChanged.connect(self.on_class_changed)
        self.btnNext.clicked.connect(self.next_image)
        self.btn_upload.clicked.connect(self.upload_image)
        self.btn_proses.clicked.connect(self.run_enhance)
        self.btn_histogram.clicked.connect(self.show_histogram)
        self.btnSalt.clicked.connect(self.run_noise)
        self.btn_jalankan_filter.clicked.connect(self.run_filter)
        self.btnPsnr.clicked.connect(self.run_psnr)
        self.btn_segmentasi.clicked.connect(self.run_segmentation)
        self.tabWidget.currentChanged.connect(lambda _: QTimer.singleShot(0, self.refresh_all))

        self.load_dataset()
        self.on_class_changed()

    # ------------------------------------------------------------ util tampilan
    def show_image(self, label, arr):
        self._shown[label] = arr
        self._render(label)

    def clear_image(self, *labels):
        for lbl in labels:
            self._shown[lbl] = None
            self._render(lbl)

    def _render(self, label):
        arr = self._shown.get(label)
        if arr is None:
            label.setPixmap(QPixmap())
            label.setText(self._placeholder.get(label, ""))
            return
        pix = QPixmap.fromImage(to_qimage(arr))
        size = label.contentsRect().size().expandedTo(label.minimumSize())
        label.setPixmap(pix.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def refresh_all(self):
        for lbl in list(self._shown):
            self._render(lbl)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_all()

    def warn(self, text):
        QMessageBox.warning(self, "Info", text)

    # ------------------------------------------------------------ dataset
    def load_dataset(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from medmnist import DermaMNIST

            print(f"Memuat DermaMNIST (split={SPLIT}, size={IMG_SIZE}) ... "
                  "pertama kali akan didownload, tunggu sebentar.")
            self.dataset = DermaMNIST(split=SPLIT, download=True, size=IMG_SIZE)
            print(f"Selesai: {len(self.dataset)} gambar.")
        except Exception as e:  # noqa: BLE001
            self.dataset = None
            QMessageBox.critical(
                self, "Gagal memuat MedMNIST",
                f"Dataset tidak bisa dimuat:\n{e}\n\n"
                "Kamu masih bisa memakai tombol 'Upload Files'.",
            )
        finally:
            QApplication.restoreOverrideCursor()

    def selected_label(self):
        code = self.comboJenisPenyakit.currentText().split("-")[0].strip().lower()
        return DERMA_LABELS.get(code)

    def on_class_changed(self, *_):
        if self.dataset is None:
            return
        label = self.selected_label()
        if label is None:
            return
        self.class_indices = np.where(self.dataset.labels.flatten() == label)[0]
        self.ptr = -1
        self.next_image()

    def next_image(self):
        if self.dataset is None:
            self.warn("Dataset belum berhasil dimuat. Gunakan 'Upload Files' atau cek koneksi internet.")
            return
        if len(self.class_indices) == 0:
            self.warn("Tidak ada gambar untuk kelas ini di split yang dipilih.")
            return
        self.ptr = (self.ptr + 1) % len(self.class_indices)
        img = self.dataset.imgs[self.class_indices[self.ptr]]
        self.set_base_image(np.ascontiguousarray(img, dtype=np.uint8))

    def upload_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih gambar", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if not path:
            return
        data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if data is None:
            self.warn("File gambar tidak bisa dibaca.")
            return
        rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > 512:  # supaya proses tetap cepat
            s = 512 / max(h, w)
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        self.set_base_image(rgb)

    def set_base_image(self, rgb):
        """Gambar baru masuk -> reset semua hasil sebelumnya."""
        self.current_rgb = rgb
        self.enhanced = self.noisy = self.filtered = self.filter_info = None

        self.show_image(self.lbl_enh_asli, rgb)
        self.show_image(self.lbl_f_asli, rgb)
        self.show_image(self.lbl_seg_asli, rgb)
        self.clear_image(self.lbl_enh_hasil, self.lbl_enh_hist,
                         self.lbl_f_noise, self.lbl_f_hasil,
                         self.lbl_seg_mask, self.lbl_seg_overlay)
        self.lineEditPsnrResult.setText("-")
        self.tabel_komparasi.setRowCount(0)

    def need_image(self):
        if self.current_rgb is None:
            self.warn("Belum ada gambar. Klik NEXT atau Upload Files dulu.")
            return False
        return True

    # ------------------------------------------------------------ tab Enhancement
    def run_enhance(self):
        if not self.need_image():
            return
        method = self.combo_metode_enhance.currentText()
        if method.startswith("Histogram"):
            self.enhanced = hist_equalization(self.current_rgb)
        elif method.startswith("AHE"):
            self.enhanced = clahe_enhance(self.current_rgb)
        else:
            self.enhanced = contrast_stretching(self.current_rgb)
        self.show_image(self.lbl_enh_hasil, self.enhanced)

    def show_histogram(self):
        if not self.need_image():
            return
        self.show_image(self.lbl_enh_hist, render_histogram(self.current_rgb, self.enhanced))

    # ------------------------------------------------------------ tab Filter
    def run_noise(self):
        if not self.need_image():
            return
        self.noisy = add_salt_pepper(self.current_rgb, self.spin_noise.value())
        self.filtered = self.filter_info = None
        self.show_image(self.lbl_f_noise, self.noisy)
        self.clear_image(self.lbl_f_hasil)
        self.lineEditPsnrResult.setText("-")
        self.tabel_komparasi.setRowCount(0)  # noise baru -> perbandingan baru

    def read_kernel(self):
        try:
            k = int(self.input_kernel.text().strip())
        except ValueError:
            self.warn("Kernel harus berupa bilangan bulat ganjil, misalnya 3, 5, atau 7.")
            return None
        k = max(3, min(k, 31))
        if k % 2 == 0:
            k += 1
        self.input_kernel.setText(str(k))
        return k

    def run_filter(self):
        if not self.need_image():
            return
        if self.noisy is None:
            self.warn("Tambahkan noise dulu lewat tombol 'Salt N Peper Noise'.")
            return
        k = self.read_kernel()
        if k is None:
            return
        name = self.combo_jenis_filter.currentText()
        if name == "median":
            self.filtered = cv2.medianBlur(self.noisy, k)
        else:
            self.filtered = cv2.blur(self.noisy, (k, k))
        self.filter_info = (name, k)
        self.show_image(self.lbl_f_hasil, self.filtered)
        self.lineEditPsnrResult.setText("-")

    def run_psnr(self):
        if self.filtered is None:
            self.warn("Jalankan 'Apply Filter' dulu sebelum menghitung PSNR.")
            return
        mse, psnr = mse_psnr(self.current_rgb, self.filtered)
        psnr_txt = "inf" if np.isinf(psnr) else f"{psnr:.2f}"
        self.lineEditPsnrResult.setText(f"{psnr_txt} dB")

        name, k = self.filter_info
        row = None  # kombinasi filter+kernel yang sama -> update baris, bukan tambah
        for r in range(self.tabel_komparasi.rowCount()):
            if (self.tabel_komparasi.item(r, 0).text() == name
                    and self.tabel_komparasi.item(r, 1).text() == str(k)):
                row = r
                break
        if row is None:
            row = self.tabel_komparasi.rowCount()
            self.tabel_komparasi.insertRow(row)
        for col, val in enumerate([name, str(k), psnr_txt, f"{mse:.2f}"]):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.tabel_komparasi.setItem(row, col, item)

    # ------------------------------------------------------------ tab Segmentasi
    def run_segmentation(self):
        if not self.need_image():
            return
        method = self.combo_metode_seg.currentText()
        try:
            mask, overlay = segment_lesion(self.current_rgb, method)
        except cv2.error as e:
            self.warn(f"Segmentasi gagal:\n{e}")
            return
        self.show_image(self.lbl_seg_mask, mask)
        self.show_image(self.lbl_seg_overlay, overlay)


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app = QApplication(sys.argv)
    win = MainDialog()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
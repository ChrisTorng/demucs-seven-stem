# demucs-seven-stem

使用 Demucs `htdemucs_6s` 對原始混音與後續 residual 反覆分軌，並把各次分軌得到的同名 stem 疊加成最終六軌。

對第 `k` 次 pass：

```text
input_0 = original
stems_k = Demucs(input_k)
residual_k = input_k - sum(stems_k)
input_(k+1) = residual_k
```

執行 `N` 次後：

```text
accumulated_stem[name] = sum(pass_k_stem[name], k=0..N-1)
final_residual = original - sum(all accumulated stems)
```

因此在相同 sample domain 與無額外量化的理想情況下：

```text
final_residual == residual_(N-1)
```

實際檔案可能有浮點加總順序或 WAV 精度造成的極小差異；`manifest.json` 會記錄 peak 與 RMS difference。

## 預設行為

```powershell
demucs-seven-stem "D:\Music\song.wav"
```

預設：

- 總共執行 2 次分軌。
- 第一次處理原音，第二次處理第一次 residual。
- 寫出兩次結果疊加後的最終六軌。
- 不保留各 pass 的中間六軌。
- 不保留各 pass 的 residual。
- 寫出 `final_residual.wav`。
- 使用 64-bit float WAV (`DOUBLE`)。

輸出：

```text
separated-seven-stem/
└─ song/
   ├─ drums.wav
   ├─ bass.wav
   ├─ other.wav
   ├─ vocals.wav
   ├─ guitar.wav
   ├─ piano.wav
   ├─ final_residual.wav
   └─ manifest.json
```

## 指定分軌次數

總共執行 3 次：

```powershell
demucs-seven-stem song.wav --passes 3
```

每增加一次 pass，就會對前一層 residual 再做一次完整六軌推論。

舊版的 `--residual-passes N` 仍可使用，其總 pass 數為 `N + 1`，但新指令建議使用 `--passes`。

## 輸出保留選項

不要寫出最終累積六軌：

```powershell
demucs-seven-stem song.wav --no-accumulated-stems
```

保留每個 pass 的六軌：

```powershell
demucs-seven-stem song.wav --keep-pass-stems
```

保留每個 pass 的 residual：

```powershell
demucs-seven-stem song.wav --keep-pass-residuals
```

不要寫出 final residual：

```powershell
demucs-seven-stem song.wav --no-final-residual
```

完整保留兩次 pass 與最終輸出：

```powershell
demucs-seven-stem song.wav `
  --passes 2 `
  --keep-pass-stems `
  --keep-pass-residuals
```

中間檔案會放在：

```text
song/pass_00/
song/pass_01/
```

## Manifest

最上層 `manifest.json` 記錄：

- 實際總 pass 數與完整設定。
- 每個 pass 的輸入、六軌與 residual peak／RMS。
- 最終累積六軌的統計。
- final residual 統計。
- 最終六軌加 final residual 的重建誤差。
- final residual 與最後一個 pass residual 的 peak／RMS difference。

## 精度

預設 `--wav-subtype DOUBLE` 使用 64-bit IEEE float WAV。這能最大限度保留：

```text
original ≈ accumulated six stems + final residual
```

節省空間可使用：

```powershell
demucs-seven-stem song.wav --wav-subtype FLOAT
```

`FLOAT` 是 32-bit float WAV，會引入 float32 量化，因此 final residual 與最後一個 pass residual 的差異通常較 `DOUBLE` 大，但仍應非常小。

程式不會逐軌 normalize、clamp 或防 clipping；浮點 WAV 可保存超過 `±1.0` 的 sample。

## Windows 安裝

建議使用 Python 3.11，並先安裝 FFmpeg：

```powershell
git clone https://github.com/ChrisTorng/demucs-seven-stem.git
cd demucs-seven-stem
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

NVIDIA GPU 請先依 PyTorch 官方安裝頁安裝適合目前 CUDA 的 `torch` 與 `torchaudio`，再安裝本專案。

執行：

```powershell
.\.venv\Scripts\demucs-seven-stem.exe "D:\Music\song.wav" --device cuda
```

CPU：

```powershell
.\.venv\Scripts\demucs-seven-stem.exe "D:\Music\song.wav" --device cpu
```

使用本機模型目錄時，可指定模型 signature 與 repo：

```powershell
demucs-seven-stem song.wav `
  --model 5c90dfd2 `
  --model-repo "D:\Models\demucs"
```

CPU 與 GPU 使用相同模型權重。GPU 主要提升速度；因浮點運算核心與加總順序不同，結果不保證逐 sample 完全相同，但通常不構成可感知的分軌品質差異。

## 多檔加總與參考差值

原有的 sample-aligned 音訊工具仍保留。

加總多個檔案或資料夾：

```powershell
demucs-seven-stem stems-folder extra.wav --audio-output sum.wav
```

參考來源減去輸入總和：

```powershell
demucs-seven-stem stems-folder `
  --reference original.wav `
  --audio-output residual.wav
```

所有檔案必須具有完全相同的 sample rate、channel 數與 sample 數。

## 開發

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
```

## 授權

本專案採用 [MIT License](LICENSE)。Demucs、PyTorch 與其他相依套件各自適用其原有授權。

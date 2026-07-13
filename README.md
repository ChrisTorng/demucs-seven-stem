# demucs-seven-stem

使用 Demucs `htdemucs_6s` 產生六個模型 stem，並額外建立第七軌
`residual.wav`：

```text
residual = pass input - (drums + bass + other + vocals + guitar + piano)
```

同一個 pass 的七個浮點 WAV 相加，可重建該 pass 的輸入。也可以把第七軌再送進
Demucs，產生下一層的六軌加新 residual。

工具也提供不需執行 Demucs 的 sample-aligned 音訊運算：

```text
sum output        = input 1 + input 2 + ...
difference output = reference - (input 1 + input 2 + ...)
```

輸入可為多個音訊檔、資料夾，或兩者混合。

## 輸出內容

預設執行：

```powershell
demucs-seven-stem "D:\Music\song.flac"
```

輸出：

```text
separated-seven-stem/
└─ song/
   ├─ manifest.json
   └─ pass_00/
      ├─ drums.wav
      ├─ bass.wav
      ├─ other.wav
      ├─ vocals.wav
      ├─ guitar.wav
      ├─ piano.wav
      ├─ residual.wav
      └─ manifest.json
```

`pass_00` 的輸入是原始混音。`residual.wav` 是模型六軌總和與該輸入之間的差值。

執行一次 residual 再分軌：

```powershell
demucs-seven-stem "D:\Music\song.flac" --residual-passes 1
```

會再建立：

```text
pass_01/
├─ drums.wav
├─ bass.wav
├─ other.wav
├─ vocals.wav
├─ guitar.wav
├─ piano.wav
├─ residual.wav
└─ manifest.json
```

`pass_01` 的輸入就是 `pass_00/residual.wav` 的實際儲存 samples。
`--residual-passes 2` 會繼續處理 `pass_01/residual.wav`，依此類推。每增加一次 pass，
都會增加一次完整的 Demucs 推論成本。

## 音訊加總與差值

這兩種模式不載入 Demucs 模型，只使用 `soundfile` 解碼並以 float64 累加。
輸出必須是 `.wav`，且沿用 `--wav-subtype DOUBLE|FLOAT`。
程式不 normalize、不 clamp，因此輸出 sample 可以超過 `±1.0`。

### 加總多個檔案

```powershell
demucs-seven-stem `
  drums.wav bass.wav other.wav vocals.wav guitar.wav piano.wav `
  --audio-output six-stem-sum.wav
```

### 加總資料夾中的全部音訊

```powershell
demucs-seven-stem `
  "D:\Stems\pass_00" `
  --audio-output "D:\Stems\seven-track-sum.wav"
```

資料夾模式預設只讀取第一層。加入 `--recursive` 可包含子資料夾：

```powershell
demucs-seven-stem `
  "D:\AudioTree" `
  --recursive `
  --audio-output "D:\AudioTree\sum.wav"
```

可以混合多個檔案與資料夾：

```powershell
demucs-seven-stem `
  "D:\StemsA" `
  "D:\Extra\effect.wav" `
  "D:\Extra\ambience.wav" `
  --audio-output combined.wav
```

資料夾內會納入常見音訊副檔名，包括 WAV、FLAC、OGG、AIFF、CAF 與 MP3；
實際能否解碼取決於目前安裝的 `libsndfile`。輸出檔本身與 `--reference` 指定的檔案
會自動從資料夾掃描結果排除，以免被重複加入。

### 以參考來源建立差值

```powershell
demucs-seven-stem `
  drums.wav bass.wav other.wav vocals.wav guitar.wav piano.wav `
  --reference original.wav `
  --audio-output residual-from-files.wav
```

計算式：

```text
residual-from-files.wav = original.wav - sum(六個輸入檔)
```

也可以直接指定放置 stems 的資料夾：

```powershell
demucs-seven-stem `
  "D:\Stems\six-only" `
  --reference "D:\Music\original.wav" `
  --audio-output "D:\Stems\residual.wav"
```

加總與差值採嚴格 sample alignment。所有輸入以及參考檔必須具備完全相同的：

- sample rate
- channel 數
- sample 數／長度
- 起始時間對齊

程式不會自動 resample、延遲補償、time stretch、截短或補零。任何不一致都會停止並指出檔案。
這可避免在不知情的情況下產生錯位或相位錯誤的總和。

如果輸出已存在，需加入 `--overwrite`：

```powershell
demucs-seven-stem stems --audio-output sum.wav --overwrite
```

## Windows 安裝

建議使用 Python 3.11，並先安裝 FFmpeg，使 `ffmpeg.exe` 可由 `PATH` 找到。

```powershell
git clone https://github.com/ChrisTorng/demucs-seven-stem.git
cd demucs-seven-stem
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### NVIDIA GPU

先依 PyTorch 官方安裝頁選擇適合目前環境的 CUDA wheel：

<https://pytorch.org/get-started/locally/>

使用該頁產生的命令，透過虛擬環境 Python 安裝 `torch` 與 `torchaudio`，再安裝本專案：

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
```

執行時不需要啟用虛擬環境：

```powershell
.\.venv\Scripts\demucs-seven-stem.exe "D:\Music\song.flac" --device cuda
```

第一次使用 `htdemucs_6s` 時，Demucs 會下載模型權重。

## 常用命令

一次處理多個檔案：

```powershell
demucs-seven-stem song1.flac song2.wav -o "D:\Stems"
```

對第七軌再做一次七軌分解：

```powershell
demucs-seven-stem song.flac --residual-passes 1
```

增加 shift averaging：

```powershell
demucs-seven-stem song.flac --shifts 10
```

指定 GPU：

```powershell
demucs-seven-stem song.flac --device cuda:0
```

覆寫既有輸出：

```powershell
demucs-seven-stem song.flac --overwrite
```

查看全部選項：

```powershell
demucs-seven-stem --help
```

## 精確重建與 WAV 格式

預設 `--wav-subtype DOUBLE`，每個 sample 使用 64-bit IEEE 浮點數。程式會：

1. 將六個模型 stem 轉換成最終 WAV 精度。
2. 使用轉換後的六軌計算 residual。
3. 寫出七個檔案。
4. 重新讀回七軌並加總。
5. 將實際 peak reconstruction error 寫入 `manifest.json`。

這避免 Demucs CLI 預設的逐軌 rescale、clamp 或整數 PCM 量化破壞加總關係。程式不會對
任何 stem normalize 或防 clipping；浮點 WAV 可以保留超過 `±1.0` 的 sample。

使用 `--wav-subtype FLOAT` 可改成 32-bit float WAV，容量約為 DOUBLE 的一半，但檔案讀回後
通常會有極小的 float32 量化重建誤差。

> 「七軌可以重建混音」只代表 mixture consistency。它不代表樂器一定被分配到正確 stem。
> 例如吉他若完整漏到 `other`，總和仍可能正確，而 residual 不會指出這個分類錯誤。

## Residual 的解讀

`residual.wav` 可能包含：

- 六軌全部漏掉的成分。
- 六軌重複估計所需的反相抵銷成分。
- 相位、時間對齊與瞬態誤差。
- 分段、shift averaging 與模型近似造成的誤差。

因此它適合作為 reconstruction error 與補償軌，不應直接視為第七種純樂器 stem。
遞迴分解 residual 可以協助觀察殘差中的可分離結構，但後續 pass 仍可能產生 bleed 與人工雜訊。

## Manifest

每個 pass 的 `manifest.json` 記錄：

- sample rate、channel 與 sample 數。
- 各 stem 與 residual 的 peak／RMS。
- residual 相對 pass input 的 RMS dB。
- 記憶體內重建 peak error。
- 七個 WAV 實際讀回相加後的 peak error。

最上層 `manifest.json` 另記錄完整執行參數與所有 passes。

## 開發

CI 不下載 Demucs 模型，只測試 residual 計算、音訊加總／差值、WAV 精度模型與基本程式碼品質。

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
```

## 授權

本專案採用 [MIT License](LICENSE)。Demucs、PyTorch 與其他相依套件各自適用其原有授權。

# マシンガントークAI

マシンガントークＡＩは話すのが苦手な人を補助するためのものです。

音声付きのMP4動画に、話と話の間が一秒以上の空きがある部分に、会話を生成して補完します。

#ロードマップ

ＣｈａｔＧＰＴに対応予定です。現在はＲｉｎｎａ１．３Ｂを使用しています。どちらも併用できるようにしたいとは思います。

#手動インストール。Condaでの設定

0. Install Conda

curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"

bash Miniconda3.sh

2. Create a new conda environment

conda create -n machinguntalk python=3.10.9

conda activate machinguntalk

4. Install Pytorch

Linux/WSL 	NVIDIA 	pip3 install torch torchvision torchaudio

Linux 	    AMD 	  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

5. Install the web UI

git clone https://github.com/rentacka/machinguntalk

cd machinguntalk

pip install -r requirements.txt

git clone https://github.com/log1stics/voice-generator-webui

mv voice-generator-webui VGwebui

cd VGwebui

pip install -r requirements.txt

5.Run

　全部インストールできたら、いったんWSLを閉じて、
 
cd machinguntalk

conda activate machinguntalk

python machinguntalk.py


＊whisper-large-v2-ct2が必要です。

ない場合はlarge-v2を使ってください。


〇voice-generator-webuiのインストールでエラー

voice-generator-webuiはVGwebui\tts\monotonic_align\monotonic_align内に空のmonotonic_alignフォルダを作ります。

そして、VGwebui\tts\monotonic_align\monotonic_align内の方にVGwebui\tts\monotonic_align内のsetup.py関連のファイルをコピーします。

これで、python setup.py build_ext --inplace するとビルドできます。

最後に、pip install -e . でインストールします。

〇FileNotFoundError　libbitsandbytes_cuda117.so'

sudo ln -s /usr/lib/wsl/lib/libcuda.so.1 /usr/local/cuda/lib64/libcuda.soで一応解決。WSL再起動必要。

さきほどの方法で解決しない場合、Cudaインストールしてるのに動かない場合は、Cudaのパスが通てないことが原因です。

CUDA複数バージョンインストール後のシステム環境変数の変更 https://blog.kintarou.com/2021/06/25/post-1591/

参照してくださいｂ

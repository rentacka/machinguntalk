# マシンガントークAI
マシンガントークＡＩは話すのが苦手な人を補助するためのものです。

#ロードマップ

ＣｈａｔＧＰＴに対応予定です。現在はＲｉｎｎａ１．３Ｂを使用しています。どちらも併用できるようにしたいとは思います。

#手動インストール。Condaでの設定

0. Install Conda
curl -sL "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > "Miniconda3.sh"
bash Miniconda3.sh

2. Create a new conda environment
conda create -n machinguntalk python=3.10.9
conda activate machinguntalk

3. Install Pytorch
Linux/WSL 	NVIDIA 	pip3 install torch torchvision torchaudio
Linux 	    AMD 	  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.4.2

4. Install the web UI
git clone https://github.com/rentacka/machinguntalk
cd machinguntalk
pip install -r requirements.txt

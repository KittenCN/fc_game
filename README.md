# FC Emulator Toolkit

璇ラ」鐩彁渚涗竴涓熀浜?Python 鐨?FC (NES) 妯℃嫙鍣ㄥ伐鍏烽泦锛屾柟渚垮姞杞?`.nes` ROM銆佹墜鍔ㄦ父鐜╋紝鍚屾椂涓哄己鍖栧涔犺嚜鍔ㄥ寲璁粌鎻愪緵鎺ュ彛銆傛牳蹇冨熀浜?[`nes-py`](https://github.com/Kautenja/nes-py)锛屽苟棰濆灏佽浜嗘帶鍒跺櫒杈撳叆銆佸浘鍍忔覆鏌撱€佺幆澧冧氦浜掔瓑妯″潡锛屼娇鍚庣画寮€鍙戞櫤鑳戒綋鏇撮珮鏁堛€?

## 鍔熻兘姒傝
- 鍔犺浇 iNES 鏍煎紡鐨?ROM 骞惰繍琛屻€?
- 浣跨敤閿洏鎴栬嚜瀹氫箟杈撳叆鍣ㄨ繘琛屽疄鏃舵父鐜┿€?
- 瀵煎嚭灞忓箷鍍忕礌銆佸唴瀛樼姸鎬佺瓑瑙傛祴淇℃伅銆?
- 鎻愪緵 Gymnasium 鍏煎鐨勭幆澧冨寘瑁呭櫒锛屾柟渚胯缁?鎺ㄧ悊绁炵粡缃戠粶銆?
- 闆嗘垚 Stable-Baselines3 璁粌鑴氭湰锛屽彲涓€閿惎鍔?PPO/A2C 璁粌銆?

## 蹇€熷紑濮?

```bash
pip install -e .
python -m fc_emulator.cli --rom path/to/game.nes
```

榛樿鎺у埗閿綅锛?
- 鏂瑰悜閿細`WASD`
- `A` 閿細`J`
- `B` 閿細`K`
- `Start`锛歚Enter`
- `Select`锛歚Right Shift`

## 寮哄寲瀛︿範璁粌涓庢帹鐞?
1. 瀹夎棰濆渚濊禆锛?
   ```bash
   pip install -e .[rl]
   ```
   鑻ヤ箣鍓嶅凡瀹夎杩囷紝鍙崟鐙ˉ鍏呭浘鍍忕缉鏀句緷璧栵細`pip install pillow`銆?
2. 浣跨敤鍐呯疆璁粌鑴氭湰锛堥粯璁や娇鐢ㄧ鏁ｅ寲鍔ㄤ綔闆嗗悎涓庣伆搴﹁娴嬶級锛?
   ```bash
   python -m fc_emulator.train --rom roms/SuperMarioBros.nes \
       --algo ppo --total-timesteps 1000000 --tensorboard
   ```
   - `--action-set` 鍙€夋嫨 `default`/`simple`锛屾垨鑷畾涔夋寜閿粍鍚堬紝渚嬪锛歚"RIGHT;A,RIGHT;B"`銆?
   - `--frame-skip` / `--frame-stack` 鎺у埗瀛愰噰鏍蜂笌鐘舵€佸爢鍙狅紝鍏煎 Stable-Baselines3 鐨?`CnnPolicy`銆?
   - `--resize HEIGHT WIDTH` 鍙互鍦ㄨ繘鍏ョ瓥鐣ョ綉缁滃墠瀵硅娴嬪浘鍍忎笅閲囨牱锛堝 `--resize 84 84`锛夛紝鏄捐憲闄嶄綆璁＄畻閲忓苟鍔犻€熻缁冦€?
   - `--vec-env subproc` 寮哄埗浣跨敤澶氳繘绋嬮噰鏍凤紝浣?CPU 澶氭牳鏇村厖鍒嗭紝榛樿 `auto` 浼氬湪 `--num-envs>1` 鏃惰嚜鍔ㄥ惎鐢ㄣ€?
   - 榛樿浼氬湪 reset 鍚庤嚜鍔ㄦ寜涓?START 鎸夐挳璺宠繃鏍囬鐣岄潰锛屽彲鐢?--disable-auto-start / --auto-start-max-frames / --auto-start-press-frames 璋冩暣
   - 鍋滄粸妫€娴嬶細榛樿鍦ㄧ害 900 甯э紙绾?15 绉掞級鍐呮棤 1 鍍忕礌鍓嶈繘灏辫Е鍙戯紝鍙€氳繃 --stagnation-frames / --stagnation-progress 璋冩暣
   - 鏀寔 蔚-greedy 鎺㈢储锛屽彲閫氳繃 --exploration-epsilon / --exploration-final-epsilon / --exploration-decay-steps 璋冩暣闅忔満鍔ㄤ綔姒傜巼
   - 澧炲姞 `--num-envs` 鍙互骞惰杩愯澶氫釜鐜锛岃繘涓€姝ュ帇姒?CPU/GPU 鍚炲悙銆?
3. 杞藉叆宸茶缁冩ā鍨嬪苟瀹炴椂鎺ㄧ悊锛?
   ```bash
   python -m fc_emulator.infer --rom roms/SuperMarioBros.nes \
       --model runs/ppo_agent_20250101-120000.zip --deterministic
   ```
   鎺ㄧ悊鏃朵細鑷姩寮€鍚?`render_mode="human"`锛屽彲瑙傚療鏅鸿兘浣撴搷浣滆繃绋嬨€傝嫢璁粌鏃朵娇鐢ㄤ簡涓嬮噰鏍凤紝璇蜂紶鍏ュ悓鏍风殑 `--resize` 鍙傛暟淇濇寔杈撳叆灏哄涓€鑷淬€?

## 闈㈠悜鑷姩鍖栬缁?
- `fc_emulator.rl_env.NESGymEnv` 鍩轰簬 `gymnasium.Env`锛屾毚闇?`step/reset/render` 鎺ュ彛銆?
- `fc_emulator.wrappers.DiscreteActionWrapper` 灏?8 涓寜閿槧灏勫埌鍙厤缃殑鏈夐檺鍔ㄤ綔闆嗗悎銆?
- `fc_emulator.train` / `fc_emulator.infer` 灏佽浜?Stable-Baselines3 鐨勫悜閲忕幆澧冦€佸抚鍫嗗彔涓庢ā鍨嬩繚瀛?鍔犺浇娴佺▼銆?
- 鍙€氳繃鑷畾涔夊洖璋冦€佸鍔卞嚱鏁版垨瑙傚療绌洪棿鎵╁睍锛屽揩閫熸帴鍏ュ叾瀹冪畻娉曞簱銆?

## 椤圭洰缁撴瀯
- `fc_emulator/`
  - `rom.py`: ROM 瑙ｆ瀽涓庡厓鏁版嵁銆?
  - `bus.py`: 鍐呭瓨鏄犲皠涓庤澶囨€荤嚎銆?
  - `controller.py`: 鎺у埗鍣ㄨ緭鍏ユ槧灏勩€?
  - `renderer.py`: Pygame 娓叉煋鍜岀獥鍙ｇ鐞嗐€?
  - `emulator.py`: 瀵?`nes_py.NESEnv` 鐨勯珮绾у皝瑁呫€?
  - `rl_env.py`: Gymnasium 鐜鍖呰鍣ㄣ€?
  - `wrappers.py`: 寮哄寲瀛︿範鍔ㄤ綔闆嗗悎涓庣鏁ｅ寲灏佽銆?
  - `rl_utils.py`: 璁粌/鎺ㄧ悊鍏辩敤鐨勭幆澧冩瀯寤哄伐鍏枫€?
  - `analysis.py`: 璁粌鏃ュ織 (`episode_log.jsonl`) 鐨勭粺璁″垎鏋?CLI銆?
  - `train.py`: 璁粌 CLI锛圥PO/A2C锛夈€?
  - `infer.py`: 鎺ㄧ悊 CLI銆?
  - `cli.py`: 鍛戒护琛屽叆鍙ｏ紝鏀寔浜虹被娓哥帺銆?
- `examples/`
  - `human_play.py`: 鎵嬪姩娓哥帺鐨勭ず渚嬨€?
  - `random_agent.py`: 闅忔満绛栫暐婕旂ず銆?

## 褰撳墠闂涓庢帰绱㈣褰?
- 瑙傚療锛歚runs/episode_log.jsonl` 涓害 1.6 涓囨潯鏁版嵁锛宍mario_x` 鍧囧€肩害 370锛屼笖鍦?722 闄勮繎鍑虹幇 400+ 娆″仠婊烇紝璇存槑渚濇棫闅句互璺ㄨ繃绗竴鏍圭閬撱€?
- 鐚滄祴锛氶殰纰嶅墠缂轰箯鎸佺画鍔╄窇+璧疯烦绫诲畯鍔ㄤ綔锛涘仠婊炴椂鎺㈢储寮哄害涓嶅锛屽鑷存櫤鑳戒綋鍙嶅灏忓箙鎶栧姩銆?
- 宸插皾璇曪細
  * 寮曞叆鍒嗘瀹忓姩浣滐紙闀挎椂闂?RUN+JUMP銆佺煭璺宠繛鍑汇€佹寔缁笅韫诧級骞跺湪楂樺仠婊炴椂浼樺厛閲囨牱銆?
  * 鍋滄粸瑙﹀彂鍚庡己鍒舵彁鍗?epsilon锛屾渶闀夸笁鍊嶉槇鍊兼椂鐩存帴灏嗛殢鏈烘鐜囨媺婊°€?
  * 濂栧姳鍑芥暟鍔犲叆鍋滄粸绐佺牬濂栧姳銆佸井灏忓墠杩涙縺鍔憋紝浠ュ強鍔ㄥ姏鐘舵€佹彁鍗囧姞鍒嗐€?
- 褰撳墠鏁堟灉锛氳緝闀垮畯鍔ㄤ綔鑳藉伓灏旇法瓒?900+锛屼絾鎬讳綋鍒嗗竷浠嶅亸鍚?300-700 鍖洪棿锛岄渶瑕佽繘涓€姝ヨ窡杩涖€?

## 鏈€鏂颁紭鍖?
- `EpsilonRandomActionWrapper` 鏍规嵁 `mario_x` 鍋滄粸鐑偣鍔ㄦ€佽皟鏁翠紭鍏堝畯鍔ㄤ綔锛屾洿鏃╄Е鍙戦暱璺濈鍔╄窇+璺宠穬缁勫悎锛屽苟鍦ㄧ獊鐮村悗杩涜鍐峰嵈锛屽噺杞荤涓€鏍圭閬撳鐨勫崱椤裤€?
- 鏂板 `python -m fc_emulator.analysis <episode_log.jsonl>` 鍛戒护锛屽彲蹇€熺粺璁″潎鍊?涓綅鏁般€佸仠婊炲抚鏁颁互鍙婄儹鐐瑰尯闂达紝渚夸簬鍦ㄨ缁冩湡闂磋拷韪瓥鐣ョ摱棰堛€?

## 鏈潵璁炬兂涓庡紑鍙戣鍒?
- 璁粌鐩戞帶锛氬鍔犲 `stagnation_frames`銆佸畯鍔ㄤ綔瑙﹀彂娆℃暟鐨?TensorBoard 缁熻锛屼究浜庡畾浣嶇瓥鐣ラ€€鍖栥€?
- 鍔ㄦ€佹帰绱細鑰冭檻缁撳悎 learnable exploration锛堜緥濡傚熀浜庣姸鎬佷环鍊煎亸绉荤殑 entropy 璋冩暣锛夋垨浣跨敤 ICM 鎻愪緵鍐呭湪濂栧姳銆?
- 绛栫暐缁撴瀯锛氬皾璇曞紩鍏?LSTM/GRU 鐗瑰緛鎻愬彇锛屼互澧炲己瀵归暱鏈熷仠婊炰俊鍙风殑璁板繂鑳藉姏銆?
- 鏁版嵁鍐嶅埄鐢細灏嗘垚鍔熻法瓒婇殰纰嶇殑杞ㄨ抗淇濆瓨涓虹绾挎暟鎹紝鐢ㄤ簬琛屼负鍏嬮殕鎴栧鍔卞缓妯°€?
- 宸ュ叿閾撅細鏋勫缓涓€浠借瘎浼拌剼鏈紝鑷姩鍥炴斁鏈€杩?N 涓?episode 鐨勭儹鐐圭墖娈碉紝杈呭姪浜哄伐鍒嗘瀽銆?

## TODO
- 淇濆瓨/璇诲彇鐘舵€侊紙save state锛夈€?
- 缃戠粶鑱旀満瀵规垬銆?
- 鏇寸簿缁嗙殑甯х巼鍚屾涓庨煶棰戣緭鍑恒€?
- 澧炲姞鏇村鍔ㄤ綔棰勮涓庤缁冩洸绾垮彲瑙嗗寲宸ュ叿銆?

娆㈣繋鏍规嵁闇€瑕佹墿灞曪紝鐢ㄤ簬绁炵粡缃戠粶璁粌鎴栧叾瀹冭嚜鍔ㄥ寲椤圭洰銆?

## 鏈€鏂板垎鏋愪笌浼樺寲锛?025-09-27锛?- **鐡堕璇婃柇**锛氬熀浜?
uns/episode_log.jsonl 鍒嗘瀽锛屽崟涓€鍥惧儚杈撳叆閰嶅悎娴呭眰 CNN 鍦ㄨ緝闀垮叧鍗′笅鏃犳硶鎹曟崏 RAM 缁嗚妭涓庨暱鏈熶緷璧栵紝绛栫暐缃戠粶鍦?900 甯у乏鍙抽绻佸仠婊炪€?- **瑙傛祴澧炲己**锛?-observation-type 鏂板 
gb_ram / gray_ram锛屽唴閮ㄤ娇鐢?ResizeObservationWrapper + 鏂扮殑 VecTransposePixelsDictWrapper/VecFrameStackPixelsDictWrapper 灏嗗浘鍍忎笌 RAM 缁勫悎鎴愬瓧鍏歌娴嬶紝骞跺湪 MarioDualFeatureExtractor 涓仈鍚堝嵎绉?+ RAM MLP 缂栫爜銆?- **缃戠粶鎵╁睍**锛歅OLICY_PRESETS 澧炲姞 mario_dual / mario_dual_large / mario_dual_lstm 绛夐璁撅紝鍙洿鎺ュ湪 CLI 涓€氳繃 --policy-preset 鍒囨崲锛涘悓鏃舵敮鎸?--algo rppo锛坰b3-contrib RecurrentPPO锛変互鍚敤 LSTM 绛栫暐銆?- **鎺㈢储寮哄寲**锛氬疄鐜?c_emulator.icm.ICMVecEnvWrapper锛岄€氳繃 --icm 涓?--icm-* 鍙傛暟鍗冲彲鍦ㄥ悜閲忕幆澧冧笂闄勫姞 ICM 鍐呭湪濂栧姳锛堥粯璁ゅ鍥惧儚鍋氶€愬抚鍗风Н缂栫爜锛屽姩鎬佹洿鏂?forward/inverse 缃戠粶骞跺啓鍏?diagnostics.intrinsic_reward锛夈€?- **璁粌鑴氭墜鏋?*锛歮ake_vector_env 鐜扮粺涓€澶勭悊杞疆銆佸抚鍫嗗彔銆両CM 鍖呰锛?rain.py / infer.py 涓嶅啀鏄惧紡璋冪敤 VecTransposeImage锛孋LI 浼氬湪瑙傛祴绫诲瀷涓庣瓥鐣ヤ箣闂村仛鍚堟硶鎬ф牎楠岋紝閬垮厤 Dict 瑙傛祴璇厤 CnnPolicy銆?
### 鎺ㄨ崘浣跨敤鏂瑰紡
1. 鑾峰彇澶氭ā鎬佽娴嬶細--observation-type rgb_ram --policy-preset mario_dual锛屽彲棰濆鍙犲姞 --resize 84 84銆?-frame-stack 4銆?2. 鍚敤 ICM锛氬湪浠ヤ笂閰嶇疆鍩虹涓婂鍔?--icm --icm-beta 0.2 --icm-eta 0.01 --icm-lr 1e-4锛汿ensorBoard 涓彲缁撳悎 diagnostics.intrinsic_reward 瑙傚療鎺㈢储鍥炴姤銆?3. 浣跨敤 LSTM锛氬畨瑁?sb3-contrib 鍚庨€夋嫨 --algo rppo --policy-preset mario_dual_lstm 鎴?mario_large_lstm锛岄厤鍚?6 涓苟琛岀幆澧冩晥鏋滄渶浣炽€?
### 涓嬩竴姝ヨ鍒?- 璇勪及 ICM 楂橀璁粌鐨勭ǔ瀹氭€э紙鍏虫敞澶栧湪濂栧姳鏄惁琚█閲婏級锛屽繀瑕佹椂瀵?eta銆?eta 鍋氬垎娈佃皟搴︺€?- 鍦ㄥ妯℃€佽缃笅鏀堕泦鏂扮殑 episode 鏃ュ織锛屽姣?RAM 鐗瑰緛璐＄尞锛堜緥濡傜粺璁?power-up銆亀orld/stage 鍒囨崲瀵瑰仠婊炶鏁扮殑褰卞搷锛夈€?- 鎺㈢储鍩轰簬 episodic memory 鐨勯檮鍔犳帰绱俊鍙凤紝鎴栦笌 count-based 濂栧姳娣峰悎锛屽噺灏?ICM 瀵?forward 妯″瀷鐨勮繃鎷熷悎椋庨櫓銆?- 杩涗竴姝ュ彲瑙嗗寲 LSTM 鍐呴儴鐘舵€侊紝璇勪及鍏跺闀挎椂闂村仠婊烇紙900+ 甯э級鍦烘櫙鐨勮蹇嗘晥鏋滐紝涓哄悗缁紩鍏ユ敞鎰忓姏鎴?Transformer 濂犲畾鍩虹銆?
# AGENTS.md — forma-veridica

## Что это

Rust-workspace для стерео-калибровки двух камер по ChArUco-доске и поточной
3D-реконструкции сцены (облака точек по кадрам видео). GUI на eframe/egui,
видео через `ffmpeg-next`, SIFT через `sift-wgpu` (GPU), трекинг `optical-flow-lk`,
solver — `vision-calibration 0.7`, матчинг — `kiddo` k-d tree.

## Структура

| Крейт | Роль |
|---|---|
| `lib_cv` | Вся CV-логика: детектор ChArUco (`calibration/charuco.rs`), калибровка (`calibration.rs`), реконструкция (`reconstruction.rs`), видеоплеер (`video.rs`) |
| `lib_pipeline` | Оркестрация: `run_calibration` / `run_reconstruction` (`src/runner.rs`), YAML-конфиги (`src/config.rs`) |
| `calibration_app` | GUI калибровки (тонкий клиент над lib_pipeline) |
| `reconstruction_app` | GUI реконструкции |
| `lib_ui` | Мелкие UI-утилиты |
| `generate_calibration_pattern` | ЛЕГАСИ: закомментирован из `members`, импортирует opencv без объявления зависимости — не собирается. Не добавлять в сборку без решения по пункту 5.5 трекера |

## Текущая цель

**Фикс багов** из `docs/PIPELINE_ISSUES.md` в порядке приоритета P0 → P1 → P2.
Работа идёт на ветке `bug-fix-run`. Один пункт ≈ один PR/коммит; групповые строки
(`11.2+11.3+11.6+11.1`, `12.1–12.3`, `17.4+15.2`) — одна работа целиком.

Внутри связок ориентируйся на комментарии строк трекера: например в P0-связке
matching первичен 11.3 (необратимая потеря пар через knn_k=10), для калибровочных
гейтов первопричина — 8.1.

## Документы

- `docs/PIPELINE_ISSUES.md` — **живой трекер, единственный источник правды о статусах**.
  Правила работы — в шапке файла: статус ⬜→🔧→✅/🚫, при ✅ строка в «Логе исправлений»
  (дата, ID, что сделано, commit), при изменении статусов синхронно обновляется сводка.
  Не забывай обновлять evidence-ссылки (файл:строка) в строках, которые чинил —
  номера строк кода плывут.
- `docs/PIPELINE_AUDIT.md` — **замороженный аудит-снапшот (2026-07-20). НЕ редактировать.**
  Проблемы идентифицируются его номерами (`7.1`, `6.5`, …); `N1…` — баги, найденные
  после аудита. Внимание: не все выводы аудита верны — см. пометки в трекере
  (например, 24.5/6.11 опровержение rotation оказалось ошибочным, 6.11 поднят до MED-HIGH).

## Сборка и валидация

```sh
cargo build                                  # workspace (generate_calibration_pattern игнорируется)
cargo test -p lib_cv                         # unit-тесты (пока только в lib_cv/src/calibration.rs)
cargo clippy --workspace 2>/dev/null         # lints; конфигов fmt/clippy нет — стиль по соседям
```

- Для сборки нужны системные библиотеки FFmpeg с заголовками: крейт `ffmpeg-next 9`
  (через `ffmpeg-sys-next`) линкует `libavcodec`/`libavformat`/`libavutil`/`libswscale`.
  На Arch/CachyOS достаточно обычного пакета `ffmpeg`; на Debian-подобных —
  пакеты `libav*-dev`.
- `sift-wgpu` считает SIFT на GPU через wgpu compute-шейдеры: при создании детектора
  выполняется `request_adapter` (`gpu_sift.rs`), и в headless-сессии без GPU/Vulkan
  (включая software-фолбэк lavapipe) инициализация падает. Поэтому пайплайн
  реконструкции в агентской сессии не запускать; unit-тесты wgpu не трогают.
- GUI-приложения в сессии агента не запускать (нужен дисплей); валидация изменений —
  статический анализ + `cargo test` + `cargo clippy`.
- Тестовое покрытие минимально: при фиксе математических узлов (DLT, homographies,
  undistortion) добавляй unit-тест в тот же файл — это явно приветствуется.

## Конвенции

- **Коммиты**: короткий английский императив в subject (≤50 симв.), по стилю истории:
  `Fix log spam`, `Replace video-rs with ffmpeg-next for video decoding`. Тело только
  если несёт пользу. В сообщении можно ссылаться на ID трекера, например
  `Fix cheirality in DLT triangulation (13.1)`.
- **Язык кода/комментариев**: комментарии в коде в основном на русском — соблюдай
  существующий стиль.
- Не мутируй git-метаданные и не коммить без запроса пользователя.

## Архитектурная карта (чтобы не перечитывать всё)

**Калибровка** (`runner.rs::run_calibration`): детекция маркеров (двухпороговый
marker-first путь, `detect_aruco_markers`) → `update_rigs` + `update_correspondes_views`
(детект на одних кадрах выполняется дважды — N6) → per-camera intrinsics
(фильтр `max_reproj_error`=2px) → rig solver: сиды → re-opt на сырых rig views (8.1)
→ BA с фиксированными интринсиками. Экспорт: K + BrownConrady5 + `cam_se3_rig`
(T_C_R: rig→camera) + `rig_se3_target`.

**Реконструкция** (`runner.rs::run_reconstruction`): SIFT один раз на кадре 0 →
`match_with_epipolar_constraint` (RootSIFT → kiddo `SquaredEuclidean` → knn_k=10 →
эпиполярный гейт) → цикл LK по **искажённым** координатам → undistort → эпиполярный
фильтр → DLT по **undistorted** пикселям → PLY `frame_NNNN.ply`. Не смешивать
distorted/undistorted: `undistort_points` возвращает пиксели (точка уже делена на z,
`backproject_pixel` сам делит).

## Грабли

- **Данные вне репо**: `/home/watermelon0guy/Видео/Experiments/` (эксп. exp_1…exp_4,
  актуален exp_2). Видео 1280×800@120fps MJPEG AVI с дефицитом пакетов ~14% (scene) /
  ~34% (board) и PTS-гэпами до 200 мс.
- **`serde_yml` не поддерживает последовательности >65 536 элементов** — не убирай
  компактизацию в `save_calibration_to_yaml` (`per_feature_residuals`/`image_manifest`
  обнуляются намеренно, 9.2).
- **Seek/PTS**: `VideoPlayer` хранит requested state, PTS декодированных кадров не
  читается, `rewind_forward` — контейерный seek на каждом шаге (7.1, P0). При правках
  `video.rs` не усугубляй: границы seek — полуинтервал `[0, duration)`, слайдер шире.
- **Порог rоста**: при изменении формул матчинга помни, что kiddo возвращает квадраты
  расстояний (Lowe-ratio 0.75 на квадратах = 0.866 по расстояниям, 11.1).
- Мёртвый код `match_first_camera_features_to_all` (`reconstruction.rs`, pub) никем
  не вызывается — живой путь только `match_with_epipolar_constraint`.

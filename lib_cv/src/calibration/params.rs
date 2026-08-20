use serde::{Deserialize, Serialize};

/// Параметры детекции ArUco-маркеров и интерполяции углов ChArUco.
///
/// Значения по умолчанию повторяют ранее захардкоженные константы.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionParams {
    /// Отступ от границы кадра: маркеры, касающиеся края ближе этого
    /// расстояния, отбрасываются (px).
    pub border_px: f32,
    /// Нижняя граница периметра маркера относительно медианного.
    pub min_size_rel: f32,
    /// Верхняя граница периметра маркера относительно медианного.
    pub max_size_rel: f32,
    /// Размер окна cornerSubPix для углов маркеров (px).
    pub refine_window_px: u32,
    /// Максимальное число итераций cornerSubPix.
    pub refine_iterations: usize,
    /// Критерий остановки cornerSubPix.
    pub refine_epsilon: f32,
    /// Минимальная «чистота границы» маркера для принятия декодирования.
    pub min_border_score: f32,
    /// Минимальное число маркеров, по которым интерполируется угол ChArUco.
    pub min_markers_per_corner: usize,
    /// Отступ от границы кадра для интерполированных углов (px).
    pub corner_border_margin_px: f32,
}

impl Default for DetectionParams {
    fn default() -> Self {
        Self {
            border_px: 3.0,
            min_size_rel: 0.6,
            max_size_rel: 1.4,
            refine_window_px: 5,
            refine_iterations: 30,
            refine_epsilon: 0.1,
            min_border_score: 0.75,
            min_markers_per_corner: 2,
            corner_border_margin_px: 2.0,
        }
    }
}

/// Параметры отбора кадров в калибровочный датасет (admission criteria).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetParams {
    /// Минимальное число углов ChArUco на камеру, чтобы детекция попала
    /// в датасет калибровки.
    pub min_corners_per_view: usize,
    /// Минимальное число камер с валидной детекцией, чтобы кадр попал
    /// в rig-датасет.
    pub min_cameras_per_frame: usize,
}

impl Default for DatasetParams {
    fn default() -> Self {
        Self {
            min_corners_per_view: 8,
            min_cameras_per_frame: 2,
        }
    }
}

/// Параметры solver'а индивидуальной калибровки камеры.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverParams {
    /// Порог reproj-ошибки (px): точки с ошибкой выше отбрасываются.
    pub max_reproj_error: f64,
    /// Минимальное число точек на view после фильтрации.
    pub min_points_per_view: usize,
    /// Удалять view целиком, если после фильтрации в нём осталось меньше
    /// `min_points_per_view` точек.
    pub remove_sparse_views: bool,
}

impl Default for SolverParams {
    fn default() -> Self {
        Self {
            max_reproj_error: 2.0,
            min_points_per_view: 4,
            remove_sparse_views: true,
        }
    }
}

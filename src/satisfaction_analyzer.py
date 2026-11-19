# -*- coding: utf-8 -*-
"""
Analyse optimisée de satisfaction client (flux vidéo)
Sorties :
 - Vidéo allégée avec bandeau texte (output_video_path)
 - Courbe PNG (output_plot_path)
 - Moyenne + verdict imprimés et renvoyés
Optimisations :
 - échantillonnage temporel (sample_fps)
 - résolution réduite (resize_width)
 - modèle MediaPipe léger (model_complexity=0)
 - annotation landmarks optionnelle (draw_landmarks=False)
"""

import cv2
import mediapipe as mp
import numpy as np
import math
import os
from collections import deque
import matplotlib.pyplot as plt

class FastSatisfactionConfig:
    WEIGHTS = {
        'posture_shoulders': 0.35,
        'head_movement': 0.25,
        'arms_position': 0.20,
        'body_lean': 0.20
    }
    # thresholds (identiques à ton code, modifiables)
    SHOULDER_ANGLE_GOOD = 165
    SHOULDER_ANGLE_BAD = 145
    HEAD_MOVEMENT_CALM = 0.02
    HEAD_MOVEMENT_AGITATED = 0.08
    SMOOTHING_WINDOW = 10  # un peu plus serré pour moins de latence
    COLOR_SATISFIED = (0, 255, 0)
    COLOR_NEUTRAL = (0, 165, 255)
    COLOR_UNSATISFIED = (0, 0, 255)

class FastBodyLanguageAnalyzer:
    """Version allégée de l'analyseur (retourne métriques sans dessiner si demandé)."""
    def __init__(self, config=None, model_complexity=0,
                 min_detection_confidence=0.3, min_tracking_confidence=0.3):
        self.config = config or FastSatisfactionConfig()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.history = {
            'shoulder_angles': deque(maxlen=self.config.SMOOTHING_WINDOW),
            'head_positions': deque(maxlen=self.config.SMOOTHING_WINDOW),
            'arm_openness': deque(maxlen=self.config.SMOOTHING_WINDOW),
            'body_lean': deque(maxlen=self.config.SMOOTHING_WINDOW),
            'satisfaction_scores': deque(maxlen=self.config.SMOOTHING_WINDOW)
        }

    # ---------- utilitaires ----------
    def _calculate_angle_horizontal(self, point1, point2):
        angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
        return 180 - abs(angle)

    # ---------- analyses similaires aux tiennes, mais renvoient valeurs uniquement ----------
    def _analyze_shoulder_posture(self, landmarks):
        try:
            L = self.mp_pose.PoseLandmark
            left_sh = landmarks[L.LEFT_SHOULDER]
            right_sh = landmarks[L.RIGHT_SHOULDER]
            left_hip = landmarks[L.LEFT_HIP]
            right_hip = landmarks[L.RIGHT_HIP]

            shoulder_angle = self._calculate_angle_horizontal(
                (left_sh.x, left_sh.y),
                (right_sh.x, right_sh.y)
            )
            back_verticality = abs((left_sh.x + right_sh.x)/2 - (left_hip.x + right_hip.x)/2)

            if shoulder_angle >= self.config.SHOULDER_ANGLE_GOOD:
                score = 90 + (180 - shoulder_angle) * 2
            elif shoulder_angle >= self.config.SHOULDER_ANGLE_BAD:
                score = 50 + (shoulder_angle - self.config.SHOULDER_ANGLE_BAD) * 2
            else:
                score = max(0, shoulder_angle - 90)

            if back_verticality < 0.05:
                score += 10

            self.history['shoulder_angles'].append(shoulder_angle)
            return np.clip(score, 0, 100)
        except Exception:
            return 50

    def _analyze_head_movement(self, landmarks):
        try:
            L = self.mp_pose.PoseLandmark
            nose = landmarks[L.NOSE]
            head_pos = (nose.x, nose.y)
            self.history['head_positions'].append(head_pos)
            if len(self.history['head_positions']) < 4:
                return 50
            positions = np.array(self.history['head_positions'])
            movement_variance = np.var(positions, axis=0).sum()
            if movement_variance < self.config.HEAD_MOVEMENT_CALM:
                score = 90
            elif movement_variance < self.config.HEAD_MOVEMENT_AGITATED:
                score = 90 - (movement_variance - self.config.HEAD_MOVEMENT_CALM) * 600
            else:
                score = max(20, 50 - movement_variance * 300)
            return np.clip(score, 0, 100)
        except Exception:
            return 50

    def _analyze_arms_position(self, landmarks):
        try:
            L = self.mp_pose.PoseLandmark
            ls = landmarks[L.LEFT_SHOULDER]
            rs = landmarks[L.RIGHT_SHOULDER]
            le = landmarks[L.LEFT_ELBOW]
            re = landmarks[L.RIGHT_ELBOW]
            lw = landmarks[L.LEFT_WRIST]
            rw = landmarks[L.RIGHT_WRIST]

            elbow_distance = abs(le.x - re.x)
            shoulder_distance = abs(ls.x - rs.x)
            openness_ratio = elbow_distance / (shoulder_distance + 1e-6)
            avg_wrist_y = (lw.y + rw.y) / 2
            avg_shoulder_y = (ls.y + rs.y) / 2
            hands_height = avg_shoulder_y - avg_wrist_y

            if openness_ratio > 1.2:
                score = 85
            elif openness_ratio > 0.8:
                score = 65
            else:
                score = 35

            if hands_height > -0.1:
                score += 15

            self.history['arm_openness'].append(openness_ratio)
            return np.clip(score, 0, 100)
        except Exception:
            return 50

    def _analyze_body_lean(self, landmarks):
        try:
            L = self.mp_pose.PoseLandmark
            ls = landmarks[L.LEFT_SHOULDER]
            rs = landmarks[L.RIGHT_SHOULDER]
            lh = landmarks[L.LEFT_HIP]
            rh = landmarks[L.RIGHT_HIP]
            shoulder_center_x = (ls.x + rs.x) / 2
            hip_center_x = (lh.x + rh.x) / 2
            lean = shoulder_center_x - hip_center_x
            if 0 < lean < 0.03:
                score = 90
            elif -0.02 < lean <= 0:
                score = 75
            elif lean >= 0.03:
                score = 60
            else:
                score = max(20, 50 + lean * 1000)
            self.history['body_lean'].append(lean)
            return np.clip(score, 0, 100)
        except Exception:
            return 50

    def compute_score_from_landmarks(self, landmarks):
        """Retourne score 0-100 et métriques sans dessin"""
        shoulder = self._analyze_shoulder_posture(landmarks)
        head = self._analyze_head_movement(landmarks)
        arms = self._analyze_arms_position(landmarks)
        lean = self._analyze_body_lean(landmarks)
        score = (
            shoulder * self.config.WEIGHTS['posture_shoulders'] +
            head * self.config.WEIGHTS['head_movement'] +
            arms * self.config.WEIGHTS['arms_position'] +
            lean * self.config.WEIGHTS['body_lean']
        )
        score = float(np.clip(score, 0, 100))
        self.history['satisfaction_scores'].append(score)
        smoothed = float(np.mean(self.history['satisfaction_scores']))
        metrics = {'shoulder': shoulder, 'head': head, 'arms': arms, 'lean': lean}
        return smoothed, metrics

    def close(self):
        try:
            self.pose.close()
        except Exception:
            pass

# ---------------------------- Fonction principale ----------------------------
def analyze_satisfaction_video_fast(
    input_video_path,
    output_video_path=None,
    output_plot_path='satisfaction_timeline.png',
    sample_fps=1.0,           # combien de "scores" par seconde à calculer (ex: 1.0)
    resize_width=640,         # réduire la résolution pour accélérer l'inférence
    draw_landmarks=False,     # dessiner landmarks (lent) : par défaut False
    model_complexity=0        # MediaPipe model complexity (0=léger)
):
    """
    Analyse optimisée d'une vidéo et produit : vidéo allégée + courbe PNG + stats.
    Retour : dict { 'avg_score', 'scores', 'timestamps', 'output_video' }
    """
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Fichier introuvable: {input_video_path}")

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo")

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # calcul du pas en frames (on analysera environ sample_fps par seconde)
    sample_fps = max(0.1, float(sample_fps))
    step = max(1, int(round(orig_fps / sample_fps)))

    # writer pour output allégé
    writer = None
    if output_video_path:
        # on écrit la vidéo à la résolution réduite (resize_width)
        scale = resize_width / float(width)
        out_w = resize_width
        out_h = int(height * scale)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_video_path, fourcc, orig_fps, (out_w, out_h))

    analyzer = FastBodyLanguageAnalyzer(model_complexity=model_complexity)
    scores = []
    timestamps = []

    frame_idx = 0
    last_score = None
    last_metrics = None

    print(f"▶ Lecture: {input_video_path} | FPS origine: {orig_fps:.1f} | Total frames: {total_frames}")
    print(f"▶ Échantillonnage à ~{sample_fps:.2f} scores/sec (step={step} frames)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # réduire la résolution pour accélérer
        scale = resize_width / float(frame.shape[1])
        small = cv2.resize(frame, (resize_width, int(frame.shape[0]*scale)))

        # si on est sur une frame échantillonnée -> analyse
        if frame_idx % step == 0:
            # conversion et inference
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            results = analyzer.pose.process(rgb)
            if results.pose_landmarks:
                smoothed_score, metrics = analyzer.compute_score_from_landmarks(results.pose_landmarks.landmark)
                last_score = smoothed_score
                last_metrics = metrics
            else:
                # pas de détection -> conserver dernier score (ou neutre)
                last_score = last_score if last_score is not None else 50.0
                last_metrics = last_metrics if last_metrics is not None else {'shoulder':50,'head':50,'arms':50,'lean':50}

            # stocker résultat
            t_sec = frame_idx / orig_fps
            scores.append(last_score)
            timestamps.append(t_sec)

        # --- CRÉER bandeau texte léger pour la vidéo de sortie (mise à jour à chaque frame) ---
        if writer:
            display_frame = small.copy()
            # couleur selon dernier_score
            sc = last_score if last_score is not None else 0
            if sc >= 70:
                color = FastSatisfactionConfig.COLOR_SATISFIED
                label = "SATISFAIT"
            elif sc >= 40:
                color = FastSatisfactionConfig.COLOR_NEUTRAL
                label = "NEUTRE"
            else:
                color = FastSatisfactionConfig.COLOR_UNSATISFIED
                label = "INSATISFAIT"

            # bandeau semi-transparent en haut
            h_out, w_out = display_frame.shape[:2]
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0,0), (w_out, 90), (0,0,0), -1)
            alpha = 0.5
            cv2.addWeighted(overlay, alpha, display_frame, 1-alpha, 0, display_frame)

            # texte score
            cv2.putText(display_frame, f"{label}: {sc:.1f}%", (16, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

            # petite barre
            bar_w = int((w_out - 40) * sc / 100)
            cv2.rectangle(display_frame, (20, 55), (20+bar_w, 70), color, -1)
            cv2.rectangle(display_frame, (20, 55), (w_out-20, 70), (255,255,255), 1)

            # ajouter métriques succinctes
            if last_metrics:
                txt = f"S:{last_metrics['shoulder']:.0f} H:{last_metrics['head']:.0f} A:{last_metrics['arms']:.0f}"
                cv2.putText(display_frame, txt, (16, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

            # Optionnel : dessiner landmarks parfois (lent) -> on évite par défaut
            if draw_landmarks and (frame_idx % (step*5) == 0):
                # dessiner landmarks brièvement (rare)
                mp_draw = mp.solutions.drawing_utils
                # results variable from last analyze step may be stale; skip if None
                try:
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(display_frame, results.pose_landmarks, analyzer.mp_pose.POSE_CONNECTIONS)
                except Exception:
                    pass

            writer.write(display_frame)

        # incrément
        frame_idx += 1

    # release
    cap.release()
    if writer:
        writer.release()
    analyzer.close()

    # statistiques finales
    avg_score = float(np.mean(scores)) if scores else 0.0
    std_score = float(np.std(scores)) if scores else 0.0
    min_score = float(np.min(scores)) if scores else 0.0
    max_score = float(np.max(scores)) if scores else 0.0

    # tracer la courbe (utilisateur demandé)
    if scores:
        plt.figure(figsize=(12,4))
        plt.plot(timestamps, scores, linewidth=2, label='Score satisfaction')
        # moyenne mobile simple (longueur = max(1, len//10))
        window = max(1, int(len(scores)/10))
        if len(scores) >= window and window>1:
            ma = np.convolve(scores, np.ones(window)/window, mode='valid')
            ma_t = timestamps[window-1:]
            plt.plot(ma_t, ma, linewidth=2.5, label=f'MA ({window})')
        plt.axhspan(70,100,alpha=0.12,color='green')
        plt.axhspan(40,70,alpha=0.10,color='orange')
        plt.axhspan(0,40,alpha=0.10,color='red')
        plt.ylim(0,100)
        plt.xlabel('Temps (s)')
        plt.ylabel('Score (%)')
        plt.title('Évolution de la satisfaction (échantillonnage optimisé)')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=150)
        plt.close()

    # verdict
    if avg_score >= 70:
        verdict = "SATISFAIT"
    elif avg_score >= 40:
        verdict = "NEUTRE"
    else:
        verdict = "INSATISFAIT"

    # impression résumé
    print("────────────────────────────────────────────")
    print(f"Résultats : avg={avg_score:.1f}% | min={min_score:.1f}% | max={max_score:.1f}% | std={std_score:.1f}")
    print(f"Verdict final : {verdict}")
    if output_video_path:
        print(f"Vidéo allégée générée : {output_video_path}")
    if os.path.exists(output_plot_path):
        print(f"Courbe sauvegardée : {output_plot_path}")
    print("────────────────────────────────────────────")

    return {
        'avg_score': avg_score,
        'min_score': min_score,
        'max_score': max_score,
        'std_score': std_score,
        'scores': scores,
        'timestamps': timestamps,
        'verdict': verdict,
        'output_video': output_video_path,
        'output_plot': output_plot_path
    }

# ---------------------------- Exemple d'utilisation ----------------------------
if __name__ == "__main__":
    import sys
    # Usage : python satisfaction_analyzer.py chemin/vers/video.mp4
    INPUT = sys.argv[1] if len(sys.argv) > 1 else "video2.mp4"
    OUTPUT = "output_satisfaction_light.mp4"
    PLOT = "satisfaction_curve.png"

    results = analyze_satisfaction_video_fast(
        INPUT,
        output_video_path=OUTPUT,
        output_plot_path=PLOT,
        sample_fps=1.0,       # 1 score / seconde -> très rapide sur vidéos lentes
        resize_width=640,     # réduit coût d'inférence
        draw_landmarks=False, # éviter dessin complet (lent)
        model_complexity=0    # modèle plus léger
    )

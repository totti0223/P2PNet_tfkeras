import tensorflow as tf
from typing import Union,Dict,Tuple
import numpy as np

class P2PMAE(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(name="p2pmae", **kwargs)
        self.mae_sum = tf.Variable(0.0, dtype=tf.float32)
        self.batch_count = tf.Variable(0, dtype=tf.int32)
        self.background_class = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        preds = y_pred[..., 2:]
        ground_truth = tf.cast(y_true[..., 2], tf.int32)
        
        softmax_preds = tf.nn.softmax(preds, axis=-1)

        max_values = tf.reduce_max(softmax_preds, axis=-1)
        max_indices = tf.cast(tf.argmax(softmax_preds, axis=-1), tf.int32)
        idx = tf.where(tf.logical_and(max_indices != 0, max_values > 0.5))
        max_values = tf.gather_nd(max_values, idx)
        max_indices = tf.gather_nd(max_indices, idx)

        predicted_counts = tf.math.bincount(tf.reshape(max_indices, [-1]))[1:]  # ignore background class
        ground_truth_counts = tf.math.bincount(tf.reshape(ground_truth, [-1]))[1:]  # ignore background class
        
        mae = tf.cast(tf.keras.losses.mean_absolute_error(ground_truth_counts, predicted_counts), tf.float32)
        self.mae_sum.assign_add(mae)
        self.batch_count.assign_add(1)

    def result(self):
        return self.mae_sum / tf.cast(self.batch_count, tf.float32)

    def reset_state(self):
        self.mae_sum.assign(0.0)
        self.batch_count.assign(0)
        
        
# import tensorflow as tf

# class nAP(tf.keras.metrics.Metric):
#     def __init__(self, r=10, deltas=None, name='nAP', **kwargs):
#         super(nAP, self).__init__(name=name, **kwargs)
#         self.r = r
#         self.deltas = deltas or [0.05 * i for i in range(1, 11)]

#     def euclidean_distance(self, y_true, y_pred):
#         return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         pred_coord = y_pred[..., 2:]
#         pred_label = y_pred[..., :2]
#         gt_coord = y_true[..., 2:]
#         gt_label = y_true[..., 2]
        
        
#         ground_truth = tf.cast(y_true[..., 2], tf.int32)
#         softmax_preds = tf.nn.softmax(preds, axis=-1)
#         max_values = tf.reduce_max(softmax_preds, axis=-1)
#         max_indices = tf.cast(tf.argmax(softmax_preds, axis=-1), tf.int32)
#         idx = tf.where(tf.logical_and(max_indices != 0, max_values > 0.5))
#         max_values = tf.gather_nd(max_values, idx)
#         max_indices = tf.gather_nd(max_indices, idx)
        
#         # Compute Euclidean distance between predicted points and ground truth points
#         distances = self.euclidean_distance(y_true, y_pred)

#         # Find nearest ground truth point within radius r for each predicted point
#         min_distances = tf.reduce_min(distances, axis=1)

#         # Mark false detections where no ground truth point is within radius r
#         false_detections = tf.cast(min_distances > self.r, dtype='float32')

#         # Find the index of the nearest predicted point for each ground truth point
#         indices = tf.argmin(distances, axis=0)

#         # Keep only the prediction with the smallest distance for each ground truth point
#         unique_indices, unique_distances = tf.unique(indices)[0]
#         unique_distances = tf.gather(distances, unique_distances)

#         # Compute precision and recall for each image
#         precisions, recalls, _ = tf.keras.metrics.PrecisionRecallIOU(num_classes=1)(y_true, y_pred)

#         # Compute average precision (AP) for each image
#         aps = []
#         for i in range(y_true.shape[0]):
#             aps.append(tf.metrics.AUC(recalls[i], precisions[i]))

#         # Apply density normalization to AP values for multiple delta values
#         naps = []
#         for delta in self.deltas:
#             density_factor = tf.exp(-tf.square(unique_distances) / (2 * tf.square(delta)))
#             normalized_aps = [ap * density_factor[j] for j, ap in enumerate(aps)]
#             naps.append(tf.reduce_mean(normalized_aps))

#         # Compute average nAP over all images
#         return tf.reduce_mean(naps)

#     def result(self):
#         return self.nap

#     def reset_states(self):
#         self.nap.assign(0)
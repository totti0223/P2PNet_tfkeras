import tensorflow as tf
from typing import Union,Dict,Tuple
import numpy as np
#from losses import hungarian_matcher
# class P2PMAE(tf.keras.metrics.Metric):
#     def __init__(self, **kwargs):
#         super().__init__(name="p2pmae", **kwargs)
#         self.mae_sum = tf.Variable(0.0, dtype=tf.float32)
#         self.batch_count = tf.Variable(0, dtype=tf.int32)
#         self.background_class = 0

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         preds = y_pred[..., 2:]
#         ground_truth = tf.cast(y_true[..., 2], tf.int32)
        
#         softmax_preds = tf.nn.softmax(preds, axis=-1)

#         max_values = tf.reduce_max(softmax_preds, axis=-1)
#         max_indices = tf.cast(tf.argmax(softmax_preds, axis=-1), tf.int32)
#         idx = tf.where(tf.logical_and(max_indices != 0, max_values > 0.5))
#         max_values = tf.gather_nd(max_values, idx)
#         max_indices = tf.gather_nd(max_indices, idx)

#         predicted_counts = tf.math.bincount(tf.reshape(max_indices, [-1]))[1:]  # ignore background class
#         ground_truth_counts = tf.math.bincount(tf.reshape(ground_truth, [-1]))[1:]  # ignore background class
        
#         mae = tf.cast(tf.keras.losses.mean_absolute_error(ground_truth_counts, predicted_counts), tf.float32)
#         self.mae_sum.assign_add(mae)
#         self.batch_count.assign_add(1)

#     def result(self):
#         return self.mae_sum / tf.cast(self.batch_count, tf.float32)

#     def reset_state(self):
#         self.mae_sum.assign(0.0)
#         self.batch_count.assign(0)
        

class P2PMAE(tf.keras.metrics.Metric):
    # single class only
    def __init__(self, **kwargs):
        super().__init__(name="p2pmae", **kwargs)
        self.background_class = 0
        self.total_absolute_error = self.add_weight(name="total_absolute_error", initializer="zeros")
        self.counter = self.add_weight(name="counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight = None):
        def calculate_metric(idx):
            # get gt number
            ground_truth = gts[idx]
            # just add up the number of 1s which is the number of objects
            gt_counts = tf.reduce_sum(ground_truth)
            
            pred = preds[idx]

            #softmax logits
            softmax_preds = tf.nn.softmax(pred, axis=-1)

            # get the max value and index
            max_values = tf.reduce_max(softmax_preds, axis=-1)
            max_indices = tf.cast(tf.argmax(softmax_preds, axis=-1), tf.int32)
            
            # get the _idx of where the max indices is 1  and max value > 0.5
            _idx = tf.where(tf.logical_and(max_indices == 1, max_values > 0.5))
            # the filtered _idx length is the predicted counts
            pred_counts = tf.reduce_sum(_idx)
            pred_counts = tf.cast(pred_counts, tf.float32)

            _absolute_error = tf.abs(gt_counts - pred_counts)
            # mean_squared_error = tf.square(gt_counts - pred_counts)
            _absolute_error = tf.cast(_absolute_error, tf.float32)            
            return _absolute_error

        gts = y_true[..., 2:]
        #gts = tf.cast(gts, tf.int32)
        preds = y_pred[..., 2:]

        absolute_error = tf.map_fn(lambda i: calculate_metric(i),
                                   elems = tf.range(tf.shape(y_true)[0]),
                                   fn_output_signature = tf.float32)
        ae = tf.reduce_mean(absolute_error)

        # Update the total absolute error and counter
        self.total_absolute_error.assign_add(ae)
        self.counter.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total_absolute_error / self.counter

    def reset_state(self):
        self.total_absolute_error.assign(0)
        self.counter.assign(0)

class P2PMSE(tf.keras.metrics.Metric):
    # single class only
    def __init__(self, **kwargs):
        super().__init__(name="p2pmse", **kwargs)
        self.background_class = 0
        self.total_squared_error = self.add_weight(name="total_squared_error", initializer="zeros")
        self.counter = self.add_weight(name="counter", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight = None):
        def calculate_metric(idx):
            # get gt number
            ground_truth = gts[idx]
            # just add up the number of 1s which is the number of objects
            gt_counts = tf.reduce_sum(ground_truth)
            
            pred = preds[idx]

            #softmax logits
            softmax_preds = tf.nn.softmax(pred, axis=-1)

            # get the max value and index
            max_values = tf.reduce_max(softmax_preds, axis=-1)
            max_indices = tf.cast(tf.argmax(softmax_preds, axis=-1), tf.int32)
            
            # get the _idx of where the max indices is 1  and max value > 0.5
            _idx = tf.where(tf.logical_and(max_indices == 1, max_values > 0.5))
            # the filtered _idx length is the predicted counts
            pred_counts = tf.reduce_sum(_idx)
            pred_counts = tf.cast(pred_counts, tf.float32)

            _squared_error = tf.square(gt_counts - pred_counts)
            _squared_error = tf.cast(_squared_error, tf.float32)            
            return _squared_error

        gts = y_true[..., 2:]
        #gts = tf.cast(gts, tf.int32)
        preds = y_pred[..., 2:]

        squared_error = tf.map_fn(lambda i: calculate_metric(i),
                                   elems = tf.range(tf.shape(y_true)[0]),
                                   fn_output_signature = tf.float32)
        se = tf.reduce_mean(squared_error)

        # Update the total absolute error and counter
        self.total_squared_error.assign_add(se)
        self.counter.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))

    def result(self):
        return self.total_squared_error / self.counter

    def reset_state(self):
        self.total_squared_error.assign(0)
        self.counter.assign(0)

# class NAP(tf.keras.metrics.Metric):
#     # single class only
#     def __init__(self, **kwargs):
#         super().__init__(name="p2pmse", **kwargs)
#         self.background_class = 0
#         self.total_nAP = self.add_weight(name="total_nAP", initializer="zeros")
#         self.counter = self.add_weight(name="counter", initializer="zeros")

#     def update_state(self, y_true, y_pred, sample_weight = None):
#         def calculate_metric(idx):
#             gt_coord = y_true[idx, ..., :2]
#             gt_label = tf.cast(y_true[idx, ..., 2:], tf.int32)
#             gt_label = tf.squeeze(gt_label, axis=-1)
#             nonzero = tf.math.count_nonzero(gt_label, dtype=tf.int32)
#             gt_label = tf.slice(gt_label, [0], [nonzero])
#             gt_coord = tf.slice(gt_coord, [0, 0], [nonzero, -1])

#             pred_coord = y_pred[idx, ..., :2]
#             pred_label = y_pred[idx, ..., 2:]
#             softmax_preds = tf.nn.softmax(pred_label, axis=-1)
#             max_values = tf.reduce_max(softmax_preds, axis=-1)
#             max_indices = tf.cast(tf.argmax(softmax_preds, axis=-1), tf.int32)
#             _idx = tf.where(tf.logical_and(max_indices == 1, max_values > 0.5))
            
#             max_values = tf.gather_nd(max_values, _idx)
#             max_indices = tf.gather_nd(max_indices, _idx)

#             #pred_label = tf.gather_nd(pred_label, _idx)
#             #pseudo predict label
#             values = tf.constant([0.0, 1.0])
#             pred_label = tf.repeat(values, tf.shape(_idx)[0])
#             pred_coord = tf.gather_nd(pred_coord, _idx)

#             #hungarian matcher
#             t_indices, p_indices, t_selector, p_selector, t_points, t_class\
#                     = hungarian_matcher(gt_coord, gt_label, pred_coord, pred_label, tau1 = self.tau1, filternonzero=False)
#             gt_coord = tf.gather(gt_coord, t_indices)
#             pred_coord = tf.gather(pred_coord, p_indices)
            

#             # Compute Euclidean distance between predicted points and ground truth points
#             distances = self.euclidean_distance(gt_coord, pred_coord)

#             # Find nearest ground truth point within radius r for each predicted point
#             min_distances = tf.reduce_min(distances, axis=1)

#             # Mark false detections where no ground truth point is within radius r
#             false_detections = tf.cast(min_distances > self.r, dtype='float32')

#             # Find the index of the nearest predicted point for each ground truth point
#             indices = tf.argmin(distances, axis=0)

#             # Keep only the prediction with the smallest distance for each ground truth point
#             unique_indices, unique_distances = tf.unique(indices)[0]
#             unique_distances = tf.gather(distances, unique_distances)

#             # Compute precision and recall for each image
#             precisions, recalls, _ = tf.keras.metrics.PrecisionRecallIOU(num_classes=1)(y_true, y_pred)

#             # Compute average precision (AP) for each image
#             aps = []
#             for i in range(y_true.shape[0]):
#                 aps.append(tf.metrics.AUC(recalls[i], precisions[i]))

#             # Apply density normalization to AP values for multiple delta values
#             naps = []
#             for delta in self.deltas:
#                 density_factor = tf.exp(-tf.square(unique_distances) / (2 * tf.square(delta)))
#                 normalized_aps = [ap * density_factor[j] for j, ap in enumerate(aps)]
#                 naps.append(tf.reduce_mean(normalized_aps))

#             # Compute average nAP over all images
#             return tf.reduce_mean(naps)

#         self.total_nAP.assign_add(se)
#         self.counter.assign_add(tf.cast(tf.shape(y_true)[0], tf.float32))



#     def euclidean_distance(self, y_true, y_pred):
#         return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), axis=-1))

#     def result(self):
#         return self.nAP / self.counter

#     def reset_state(self):
#         self.total_nAP.assign(0)
#         self.counter.assign(0)


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
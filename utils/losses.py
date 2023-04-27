import tensorflow as tf
from typing import Union,Dict,Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np
from loguru import logger

import tensorflow as tf
from typing import Union,Dict,Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np

def merge(box_a: tf.Tensor, box_b: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    A = tf.shape(box_a)[0] # Number of bbox in box_a
    B = tf.shape(box_b)[0] # Number of bbox in box b
    tiled_box_a = tf.tile(tf.expand_dims(box_a, axis=1), [1, B, 1])
    tiled_box_b = tf.tile(tf.expand_dims(box_b, axis=0), [A, 1, 1])
    return tiled_box_a, tiled_box_b

def np_tf_linear_sum_assignment(matrix):
    indices = linear_sum_assignment(matrix)
    target_indices = indices[1]
    pred_indices = indices[0]
    target_selector = np.zeros(matrix.shape[1])
    target_selector[target_indices] = 1
    target_selector = target_selector.astype(bool)

    pred_selector = np.zeros(matrix.shape[0])
    pred_selector[pred_indices] = 1
    pred_selector = pred_selector.astype(bool)
    return [target_indices, pred_indices, target_selector, pred_selector]

def hungarian_matcher(t_points, t_class, p_points, p_class, tau1=0.05):
        '''
        t_points: tensor of shape [num_of_gt_boxes, 2]
        t_class: [num_of_get_boxes, 1] # last dimention is a int containing values from 0 to num_classes. 0 is background

        p_points: [num_of_pred_boxes, 2]
        t_class: [num_of_pred_boxes, 2] last dimension is one hot encoded
        tau1: weight used for matching
        '''
        # exclude the zero elements first from the GT
        t_class = tf.squeeze(t_class, axis=-1)
        nonzero = tf.math.count_nonzero(t_class, dtype=tf.int32)
        t_class = tf.slice(t_class, [0], [nonzero])
        t_points = tf.slice(t_points, [0, 0], [nonzero, -1])
        
        softmax = tf.nn.softmax(p_class)
        cost_class = -tf.gather(softmax, t_class, axis=1)

        
        _p_points, _t_points = merge(p_points, t_points)
        # l2 norm
        cost_point = tf.norm(_p_points - _t_points, ord=2, axis=-1)
        
        cost_matrix = tau1 * cost_point + 1.0 * cost_class
        selectors = tf.numpy_function(np_tf_linear_sum_assignment, [cost_matrix], [tf.int64, tf.int64, tf.bool, tf.bool] )
        target_indices = selectors[0]
        pred_indices = selectors[1]
        target_selector = selectors[2]
        pred_selector = selectors[3]

        return target_indices, pred_indices, target_selector, pred_selector, t_points, t_class

def loss_labels(p_points, p_class, t_points, t_class, t_indices, p_indices, t_selector, p_selector, background_class=0, lambda1=0.5):

    neg_indices = tf.squeeze(tf.where(p_selector == False), axis=-1)
    neg_p_class = tf.gather(p_class, neg_indices)
    neg_t_class = tf.zeros((tf.shape(neg_p_class)[0],), tf.int64) + background_class

    neg_weights = tf.zeros((tf.shape(neg_indices)[0],)) + lambda1 #+ 0.1 #
    pos_weights = tf.zeros((tf.shape(t_indices)[0],)) + 1.0
    weights = tf.concat([neg_weights, pos_weights], axis=0)

    pos_p_class = tf.gather(p_class, p_indices)
    pos_t_class = tf.gather(t_class, t_indices)

    #############
    # Metrics
    #############
    # True negative
    cls_neg_p_class = tf.argmax(neg_p_class, axis=-1)
    true_neg  = tf.reduce_mean(tf.cast(cls_neg_p_class == background_class, tf.float32))
    # True positive
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    true_pos = tf.reduce_mean(tf.cast(cls_pos_p_class != background_class, tf.float32))
    # True accuracy
    cls_pos_p_class = tf.argmax(pos_p_class, axis=-1)
    pos_accuracy = tf.reduce_mean(tf.cast(cls_pos_p_class == pos_t_class, tf.float32))

    targets = tf.concat([neg_t_class, pos_t_class], axis=0)
    preds = tf.concat([neg_p_class, pos_p_class], axis=0)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(targets, preds)
    loss = tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)
    return loss

def loss_points(lossfunc, p_points,p_class,
                 t_points, t_class, t_indices,
                   p_indices, t_selector, p_selector):
    p_points = tf.gather(p_points, p_indices)
    t_points = tf.gather(t_points, t_indices)
    loss = lossfunc(p_points, t_points)
    #loss = tf.reduce_sum(loss)
    return loss

@tf.function
def P2PLoss(y_true, y_pred):
    def calculate_loss(idx):
        # Select a single element from the batch
        p_points = predicted_points[idx]
        p_class = predicted_label[idx]
        t_points = target_points[idx]
        t_class = target_label[idx]
        
        # Match predicted and target points
        t_indices, p_indices, t_selector, p_selector, t_points, t_class\
                = hungarian_matcher(t_points, t_class, p_points, p_class)
        

        # Calculate points loss and label loss
        _points_cost = loss_points(tf.keras.losses.MeanSquaredError(),
                                    p_points, p_class, 
                                    t_points, t_class,
                                    t_indices, p_indices,
                                    t_selector, p_selector)
#             _label_cost, true_neg, true_pos, pos_accuracy = loss_labels(p_points, p_class, t_points, t_class, t_indices, p_indices, t_selector, p_selector, background_class=0)
        _label_cost = loss_labels(p_points, p_class,
                                    t_points, t_class,
                                    t_indices, p_indices,
                                    t_selector, p_selector,
                                    background_class=0)

        return _label_cost, _points_cost
    # gt_points, gt_labels = y_true[..., :2], y_true[..., 2:]
    # pred_points, pred_logits = y_pred[..., :2], y_pred[..., 2:]    
    predicted_points = y_pred[...,0:2]
    predicted_label = y_pred[...,2:]
    target_points = y_true[...,0:2]
    target_label = tf.cast(y_true[...,2:], tf.int64)
    
    # Use tf.map_fn() to calculate loss for each element in the batch
    label_cost, points_cost = tf.map_fn(
        lambda i: calculate_loss(i),
        elems=tf.range(tf.shape(predicted_points)[0]),
        fn_output_signature=(tf.float32, tf.float32)
    )

    # Sum up the losses
    label_cost = tf.reduce_sum(label_cost)
    points_cost = tf.reduce_sum(points_cost)        

    return label_cost + points_cost * 0.0002
    #return label_cost, points_cost
    # # Match the pred_points and gt_points
    # matched_pred_points, matched_pred_logits, matched_gt_points = match_points(pred_points, gt_points, pred_logits, gt_labels)

    # # Calculate the coordinate loss (MSE)
    # coord_loss = tf.keras.losses.mean_squared_error(matched_gt_points, matched_pred_points)

    # # Calculate the label loss
    # gt_labels_one_hot = tf.one_hot(gt_labels, depth=2)
    # label_loss = tf.keras.losses.categorical_crossentropy(gt_labels_one_hot, matched_pred_logits, from_logits=True)

    # # Combine the losses
    # total_loss = coord_loss + label_loss

    # return total_loss
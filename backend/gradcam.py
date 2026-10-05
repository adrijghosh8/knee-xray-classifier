import numpy as np
import tensorflow as tf
import cv2


LAST_CONV_LAYER = "conv2d_2"


def make_gradcam_heatmap(img_array, model, class_index):

    # Make sure the model has been built/called
    _ = model(img_array, training=False)

    # Get the last convolutional layer
    last_conv_layer = model.get_layer(LAST_CONV_LAYER)

    # Model that outputs the feature maps from the last conv layer
    conv_model = tf.keras.Model(
        inputs=model.inputs[0],
        outputs=last_conv_layer.output
    )

    with tf.GradientTape() as tape:

        # Get convolutional feature maps
        conv_outputs = conv_model(img_array, training=False)

        # Watch the feature maps explicitly
        tape.watch(conv_outputs)

        # Continue forward pass from the conv layer
        x = conv_outputs

        # Find the position of the conv layer
        conv_index = model.layers.index(last_conv_layer)

        # Pass through all layers after conv2d_2
        for layer in model.layers[conv_index + 1:]:
            x = layer(x)

        predictions = x

        # Score for the predicted class
        class_score = predictions[:, class_index]

    # Gradient of class score with respect to conv feature maps
    gradients = tape.gradient(
        class_score,
        conv_outputs
    )

    if gradients is None:
        raise ValueError(
            "Gradients are None. Grad-CAM could not calculate "
            "the gradient for the selected convolutional layer."
        )

    # Average gradient for every feature map
    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    conv_outputs = conv_outputs[0]
    pooled_gradients = pooled_gradients[0]

    # Weight each feature map by its importance
    heatmap = tf.reduce_sum(
        conv_outputs * pooled_gradients,
        axis=-1
    )

    # ReLU
    heatmap = tf.maximum(heatmap, 0)

    # Normalize
    max_value = tf.reduce_max(heatmap)

    if max_value > 0:
        heatmap = heatmap / max_value

    return heatmap.numpy()


def create_gradcam_overlay(original_image, heatmap, alpha=0.4):

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.resize(
        heatmap,
        (original_image.shape[1], original_image.shape[0])
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(
            original_image,
            cv2.COLOR_GRAY2BGR
        )

    original_image = np.uint8(original_image)

    overlay = cv2.addWeighted(
        original_image,
        1 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay
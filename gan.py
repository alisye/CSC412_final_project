from numpy.random import randint
import tensorflow as tf
import os

data_dir = './data'
BATCH_SIZE = 64
img_height = 64
img_width = 64

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)


train_ds = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_width, img_height),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        color_mode="grayscale")

def make_discriminator_model():
    in_label = tf.keras.layers.Input(shape=(1,))
    li = tf.keras.layers.Embedding(2, 50)(in_label)
    n_nodes = 64 * 64
    
    li = tf.keras.layers.Dense(n_nodes)(li)
    li = tf.keras.layers.Reshape((64, 64, 1))(li)
    
    in_image = tf.keras.layers.Input(shape=(64, 64, 1))
    merge = tf.keras.layers.Concatenate()([in_image, li])
    
    fe = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(merge)
    fe = tf.keras.layers.LeakyReLU()(fe)
    fe = tf.keras.layers.Dropout(0.3)(fe)

    fe = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(fe)
    fe = tf.keras.layers.LeakyReLU()(fe)
    fe = tf.keras.layers.Dropout(0.3)(fe)

    fe = tf.keras.layers.Flatten()(fe)
    out_layer = tf.keras.layers.Dense(1)(fe)
    
    model = tf.keras.models.Model([in_image, in_label], out_layer)
    return model


def make_generator_model():
    in_label = tf.keras.layers.Input(shape=(1,))
    li = tf.keras.layers.Embedding(2, 50)(in_label)
    
    n_nodes = 16 * 16
    
    li = tf.keras.layers.Dense(n_nodes)(li)
    
    li = tf.keras.layers.Reshape((16, 16, 1))(li)
    
    in_lat = tf.keras.layers.Input(shape=(100,))
    
    n_nodes = 256 * 16 * 16
    
    gen = tf.keras.layers.Dense(n_nodes)(in_lat)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.LeakyReLU()(gen)
    gen = tf.keras.layers.Reshape((16, 16, 256))(gen)
    
    merge = tf.keras.layers.Concatenate()([gen, li])
    print(merge.shape)

    gen = tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merge)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.ReLU()(gen)
    print(gen.shape)
    
    gen = tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(gen)
    gen = tf.keras.layers.BatchNormalization()(gen)
    gen = tf.keras.layers.ReLU()(gen)
    print(gen.shape)

    gen = tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False)(gen)
    out_layer = tf.keras.layers.Activation(activation='tanh')(gen)
    print(out_layer.shape)
    
    model = tf.keras.models.Model([in_lat, in_label], out_layer)
    return model

discriminator = make_discriminator_model()
generator = make_generator_model()

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


@tf.function
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_labels = tf.convert_to_tensor(randint(0, 2, BATCH_SIZE).astype('float32'))
        generated_images = generator([noise, fake_labels], training=True)
        
        real_output = discriminator([images, labels], training=True)
        fake_output = discriminator([generated_images, fake_labels], training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return real_output, fake_output, gen_loss, disc_loss

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

checkpoint.restore(checkpoint_dir + '/ckpt-10')

def train(dataset, epochs):
    epoch = 0
    for epoch in range(epochs):
        i = 0
        avg_gen_loss = 0
        avg_disc_loss = 0
        for image_batch, labels in dataset:
            i += 1
            real_output, fake_output, gen_loss, disc_loss = train_step(image_batch, labels)
            if i % 50 == 0:
                tf.print('[%d/%d][%d/%d] D(x) = %s D(G(z)) = %s G loss: %s D loss: %s' % 
                         (epoch+1, EPOCHS, i, len(dataset),
                          str(tf.math.sigmoid(real_output).numpy().mean()), 
                          str(tf.math.sigmoid(fake_output).numpy().mean()),
                          str(gen_loss.numpy()),
                          str(disc_loss.numpy())
                         ))
            if i == len(dataset):
                break
            avg_gen_loss += gen_loss.numpy()
            avg_disc_loss += disc_loss.numpy()
        # save losses every epoch for plotting
        open('./losses/gen_loss.txt', 'a').write('\n')
        open('./losses/gen_loss.txt', 'a').write(str(avg_gen_loss/len(dataset)))
        open('./losses/disc_loss.txt', 'a').write('\n')
        open('./losses/disc_loss.txt', 'a').write(str(avg_disc_loss/len(dataset)))
        # Save the model every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)
            tf.print('checkpoint created! %s epochs completed' % str(epoch+1))


EPOCHS = 100

train(train_ds, EPOCHS)

print('training complete!...')
print('saving generator!...')
generator.save_weights('./checkpoints/g_my_checkpoint')
print('saving discriminator!...')
discriminator.save_weights('./checkpoints/d_my_checkpoint')

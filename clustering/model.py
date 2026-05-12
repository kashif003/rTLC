from pl_bolts.models.autoencoders import VAE

# Example: pretrained VAE on ImageNet (encoder features can be reused)
model = VAE(input_height=64,  # your image size
            pretrained='imagenet2012')   # downloads pretrained weights

encoder = model.encoder
decoder = model.decoder

print(model)
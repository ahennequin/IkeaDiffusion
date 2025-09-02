from datetime import datetime
from diffusers import DDIMScheduler, DDPMPipeline
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from ikea_dataset import IkeaDataset


def fine_tuning_routine(
    image_pipe,
    dataset,
    optimizer,
    device: torch.device,
    dataloader_params: dict,
):
    # Training params
    epochs = 10
    grad_accumulation_steps = 2

    # Load dataset
    train_data = DataLoader(dataset, **dataloader_params)

    losses = []

    for epoch in tqdm(
        range(epochs),
        unit="epochs",
    ):

        for step, batch in tqdm(
            enumerate(train_data),
            total=len(train_data),
            unit="batches",
            leave=False,
        ):
            _, data = batch
            # Put label (clean) images on 'device'
            clean_imgs = data.to(device)
            # Sample noise to add to clean images
            noise = torch.randn(clean_imgs.shape).to(device)
            batch_size = data.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.config.num_train_timesteps,
                (batch_size,),
                device=device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = image_pipe.scheduler.add_noise(
                clean_imgs,
                noise,
                timesteps,
            )

            # Get the model prediction for the noise
            noise_pred = image_pipe.unet(
                noisy_images,
                timesteps,
                return_dict=False,
            )[0]

            # Compare the prediction with the actual noise:
            loss = torch.nn.functional.mse_loss(
                noise_pred, noise
            )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

            # Store for later plotting
            losses.append(loss.item())

            # Update the model parameters with the optimizer based on this loss
            loss.backward()

            # Gradient accumulation:
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(
            f"Epoch {epoch} average loss: {sum(losses[-len(train_data):])/len(train_data)}"
        )

        # Save fine-tuned pipeline
        image_pipe.save_pretrained(
            f"{datetime.now().strftime('%Y%m%d%H%M%S')}_finetuned_model"
        )


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    image_pipe.to(device)

    # Create new scheduler and set num inference steps
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=40)

    image_pipe.scheduler = scheduler

    images = image_pipe(num_inference_steps=40).images
    images[0]

    # Define preprocessing step
    image_size = 256
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    # Instanciate dataset
    dataset = IkeaDataset(
        "./scrapped_data.csv",
        "./dataset",
        preprocess=preprocess,
        download=False,
    )

    dataloader_params = {
        "batch_size": 4,
        "shuffle": True,
        "num_workers": 0,
        "pin_memory": True,
    }

    # Instanciate optimizer
    lr = 1e-5
    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)

    fine_tuning_routine(
        image_pipe,
        dataset,
        optimizer,
        device,
        dataloader_params,
    )

import numpy as np


def update_particle_position_and_direction(
    position,
    direction,
    global_drift,
    interaction_range,
    velocity,
    noise_magnitude,
    time_step=1,
):
    """
    Update the particle positions and directions using the Vicsek model

    Params:
        position: An array of particle positions (A)
        direction: An array of particle directions (radians)
        global_drift: The global drift (A/time)
        interaction_range: The distance at which particles interact (A)
        velocity: The constant velocity of the system (A/time)
        noise_magnitude: The direction noise magnitude (radians)

    Returns:
        (position, direction) of the particles

    """

    # The time step
    dt = time_step

    # For the noise draw random samples form a normal distribution
    nu = np.random.normal(0, noise_magnitude, size=position.shape[0])

    # Update the direction by taking the mean direction of particles within the
    # interaction range for each particle
    direction_mean = np.zeros_like(direction)
    for i in range(position.shape[0]):
        distance = np.linalg.norm(position[i] - position, axis=1)
        direction_mean[i] = np.angle(
            np.mean(np.exp(1j * direction[distance <= interaction_range]))
        )
    direction = direction_mean + nu

    # Ensure is a numpy array
    global_drift = np.array(global_drift)

    # Compute the new position of the particle
    position = position.copy()
    position[:, :2] = position[:, :2] + dt * (
        global_drift
        + velocity * np.stack([np.cos(direction), np.sin(direction)], axis=1)
    )

    # Return the position and direction
    return position, direction

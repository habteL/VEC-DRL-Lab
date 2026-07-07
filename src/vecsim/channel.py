import math


class WirelessChannel:
    def __init__(self, bandwidth, transmission_power, noise_factor):
        self.bandwidth = float(bandwidth)
        self.transmission_power = float(transmission_power)
        self.noise_factor = float(noise_factor)

    def compute_rate(self, distance):
        if distance < 1.0:
            distance = 1.0  # near-field saturation
        snr = self.transmission_power / (distance**2 + self.noise_factor)
        rate = self.bandwidth * math.log2(1 + snr)
        return rate

    def compute_delay(self, data_size, distance):
        rate = self.compute_rate(distance)
        return data_size / rate  # seconds
        
if __name__ == "__main__":
    import math
    
    STEP_DURATION = 0.1
    channel = WirelessChannel(
        bandwidth=5_000_000,      # 5 MB/s
        transmission_power=1000,
        noise_factor=1
    )
    
    for distance in [1, 5, 10, 9.9]:
        rate = channel.compute_rate(distance)
        delay_sec = channel.compute_delay(2_000_000, distance)
        delay_steps = math.ceil(delay_sec / STEP_DURATION)
        print(f"distance={distance:5.1f} | "
              f"rate={rate/1_000_000:6.2f} MB/s | "
              f"delay={delay_sec:.3f}s | "
              f"steps={delay_steps}")
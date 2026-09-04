reader = IMPORTLIBSFILE("data/dataset1/sample1_1.libs")
print(f"{reader.n_scans} scans, {len(reader.wavelengths)} pixels")
print(f"wavelength range: {reader.wavelengths.min():.2f}-{reader.wavelengths.max():.2f}")

first_batch = next(reader.iter_window(batch_size=5))
for i, spectrum in enumerate(first_batch):
    print(f"spectrum {i}: {spectrum[:5]}")

# capdata caps the total scans read; batchsize sets the default iter_window
# chunk size, so it can be called with no arguments below.
capped_reader = IMPORTLIBSFILE("data/dataset1/sample1_1.libs", capdata=12, batchsize=5)
print(f"\ncapped to {capped_reader.n_scans} scans")
for batch in capped_reader.iter_window():
    print(f"batch of {len(batch)}")

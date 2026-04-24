class DimensionCalculator:
    def __init__(self, material_strength, seismic_force):
        self.material_strength = material_strength
        self.seismic_force = seismic_force

    def calculate_core_wall(self, height, thickness):
        # Calculate optimal dimension for core wall
        stress_limit = self.material_strength - (self.seismic_force / (height * thickness))
        return max(thickness, stress_limit)

    def calculate_column(self, height, load):
        # Calculate optimal dimension for column
        stress_limit = self.material_strength - (load / height)
        return max(height/20, stress_limit)

    def calculate_beam(self, span, load):
        # Calculate optimal dimension for beam
        stress_limit = (load * span) / (self.material_strength * span)
        return max(span/30, stress_limit)

    def calculate_slab(self, span, load):
        # Calculate optimal dimension for slab
        stress_limit = (load * span) / (self.material_strength * span)
        return max(span/40, stress_limit)
import json
from pathlib import Path
from cil.framework import AcquisitionGeometry, AcquisitionData
from cil.processors import TransmissionAbsorptionConverter, CentreOfRotationCorrector, PaganinProcessor, RingRemover
from cil.recon import FDK

class PolarisPipeline:
    def __init__(self, data, delta, beta, energy):
        self.data = data
        self.sinogram = None
        self.ag = None
        self.ig = None
        self.recon = None
        self.sino_rings = None
        self.sino_pag = None
        self.reconstructed = None
        
        # Paganin variables
        self.delta = delta
        self.beta = beta
        self.energy = energy

    def get_sinogram(self):
        """
        Assumes input data is already normalised.
        """
        self.ag = self.data.geometry
        self.sinogram = AcquisitionData(self.data, geometry=self.ag)
        self.ig = self.ag.get_ImageGeometry()
        print(self.data.shape)

        # Optional voxel size overrides (kept for reference)
        # self.ig.voxel_size_x = voxel_size_mm
        # self.ig.voxel_size_y = voxel_size_mm

        self.sinogram = TransmissionAbsorptionConverter()(self.sinogram)
        print("TransmissionAbsorptionConverter done")
        

    def correct_rotation(self):
        if self.sinogram is None or self.ig is None:
            raise RuntimeError("Sinogram and image geometry must be initialised first")

        self.sinogram.reorder(order="tigre")

        processor = CentreOfRotationCorrector.image_sharpness(
            "centre",
            backend="tigre",
            tolerance=10,
        )
        
        processor.set_input(self.sinogram)
        self.sinogram = processor.get_output()
        print("Centre of Rotation correction done")

    def ring_correction(self):
        ringRemove = RingRemover()
        ringRemove.set_input(self.sinogram)
        self.sino_rings = ringRemove.get_output()
        
        self.sino_rings.array = self.sino_rings.array.astype("float32")
        self.sino_rings.geometry.dtype = 'float32'
        print("ring removal done")

    def paganin(self):
        paganin_processor = PaganinProcessor(delta = self.delta, beta = self.beta, energy = self.energy, full_retrieval= False)
        paganin_processor.set_input(self.sino_rings)

        self.sino_pag = paganin_processor.get_output()
        self.sino_pag.array = self.sino_pag.array.astype("float32")
        self.sino_pag.geometry.dtype = 'float32'

        print("paganin done")

    def reconstruct(self):
        if self.sino_pag is not None:
            fdk =  FDK(self.sino_pag, self.ig)
        else:
            fdk =  FDK(self.sino_rings, self.ig)

        self.reconstructed = fdk.run()
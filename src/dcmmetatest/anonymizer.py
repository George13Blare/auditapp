"""Модуль анонимизации DICOM-файлов с сохранением целостности исследований."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

try:
    import pydicom
    from pydicom.dataset import Dataset

    HAS_PYDICOM = True
except ImportError:
    pydicom = None  # type: ignore
    HAS_PYDICOM = False

logger = logging.getLogger(__name__)

# Теги для анонимизации (DICOM Standard PS3.15, E.1.1 Basic Application Level Confidentiality Profile)
BASIC_ANONYMIZATION_TAGS = {
    # Patient
    (0x0010, 0x0010): "PatientName",
    (0x0010, 0x0020): "PatientID",
    (0x0010, 0x0030): "PatientBirthDate",
    (0x0010, 0x0032): "PatientBirthTime",
    (0x0010, 0x0033): "PatientBirthDateInAlternativeCalendar",
    (0x0010, 0x0034): "PatientDeathDateInAlternativeCalendar",
    (0x0010, 0x0035): "PatientReligiousPreference",
    (0x0010, 0x0101): "PatientWeight",
    (0x0010, 0x1000): "OtherPatientIDs",
    (0x0010, 0x1001): "OtherPatientNames",
    (0x0010, 0x1002): "OtherPatientIDsSequence",
    (0x0010, 0x1005): "PatientBirthName",
    (0x0010, 0x1010): "PatientAge",
    (0x0010, 0x1030): "PatientSize",
    (0x0010, 0x1040): "PatientAddress",
    (0x0010, 0x1050): "InsurancePlanIdentification",
    (0x0010, 0x1060): "PatientMotherBirthName",
    (0x0010, 0x1080): "MilitaryRank",
    (0x0010, 0x1081): "BranchOfService",
    (0x0010, 0x1090): "MedicalRecordLocator",
    (0x0010, 0x2000): "MedicalRecords",
    (0x0010, 0x2110): "Allergies",
    (0x0010, 0x2150): "CountryOfResidence",
    (0x0010, 0x2152): "RegionOfResidence",
    (0x0010, 0x2154): "PhoneNumbers",
    (0x0010, 0x2155): "EmergencyPhoneNumbers",
    (0x0010, 0x2160): "EthnicGroup",
    (0x0010, 0x2180): "Occupation",
    (0x0010, 0x21A0): "SmokingStatus",
    (0x0010, 0x21B0): "AdditionalPatientHistory",
    (0x0010, 0x21C0): "PregnancyStatus",
    (0x0010, 0x21F0): "AdmissionDate",
    (0x0010, 0x21F1): "AdmissionTime",
    (0x0010, 0x21F2): "DischargeDate",
    (0x0010, 0x21F3): "DischargeTime",
    (0x0010, 0x21F4): "DischargeDiagnosisDescription",
    (0x0010, 0x21F5): "DischargeDiagnosisCodeSequence",
    (0x0010, 0x21F6): "SpecialNeeds",
    (0x0010, 0x21F7): "NursingDiagnosis",
    (0x0010, 0x21F8): "Disposition",
    (0x0010, 0x21F9): "AdmittingDiagnosesDescription",
    (0x0010, 0x21FA): "AdmittingDiagnosesCodeSequence",
    (0x0010, 0x21FC): "AdvanceDirectivesCodeSequence",
    (0x0010, 0x2201): "ImplantRegulatoryApprovalType",
    (0x0010, 0x2202): "ImplantTargetAnatomySequence",
    (0x0010, 0x2203): "ImplantLaterality",
    (0x0010, 0x2204): "ImplantGeometricApproach",
    (0x0010, 0x2205): "ImplantTargetAnatomyFreehandDescription",
    (0x0010, 0x2206): "ImplantRegulatoryApprovalAuthority",
    (0x0010, 0x2207): "ImplantRegulatoryApprovalNumber",
    (0x0010, 0x2208): "ImplantRegulatoryApprovalDate",
    (0x0010, 0x2209): "ImplantRegulatoryApprovalExpirationDate",
    (0x0010, 0x2210): "ImplantSize",
    (0x0010, 0x2211): "ImplantDiameter",
    (0x0010, 0x2212): "ImplantLength",
    # Study
    (0x0008, 0x0050): "AccessionNumber",
    (0x0008, 0x0054): "RetrieveAETitle",
    (0x0008, 0x0055): "StationAETitle",
    (0x0008, 0x0056): "InstanceAvailability",
    (0x0008, 0x0061): "Modality",
    (0x0008, 0x0080): "InstitutionName",
    (0x0008, 0x0081): "InstitutionAddress",
    (0x0008, 0x0082): "InstitutionCodeSequence",
    (0x0008, 0x0090): "ReferringPhysicianName",
    (0x0008, 0x0092): "ReferringPhysicianAddress",
    (0x0008, 0x0094): "ReferringPhysicianTelephoneNumbers",
    (0x0008, 0x0096): "ReferringPhysicianIdentificationSequence",
    (0x0008, 0x009C): "ConsultingPhysicianName",
    (0x0008, 0x009D): "ConsultingPhysicianIdentificationSequence",
    (0x0008, 0x1030): "StudyDescription",
    (0x0008, 0x1048): "PhysicianOfRecord",
    (0x0008, 0x1049): "PhysicianOfRecordIdentificationSequence",
    (0x0008, 0x1050): "PerformingPhysicianName",
    (0x0008, 0x1052): "PerformingPhysicianIdentificationSequence",
    (0x0008, 0x1060): "NameOfPhysiciansReadingStudy",
    (0x0008, 0x1062): "PhysiciansReadingStudyIdentificationSequence",
    (0x0008, 0x1070): "OperatorsName",
    (0x0008, 0x1072): "OperatorIdentificationSequence",
    (0x0008, 0x1080): "AdmittingDiagnosesDescription",
    (0x0008, 0x1084): "AdmittingDiagnosesCodeSequence",
    (0x0008, 0x1090): "ManufacturerModelName",
    (0x0008, 0x1110): "ReferencedStudySequence",
    (0x0008, 0x1120): "ReferencedPatientSequence",
    (0x0008, 0x1155): "ReferencedSOPClassUIDInFile",
    (0x0008, 0x1195): "TransactionUID",
    (0x0008, 0x1199): "SourceApplicationEntityTitle",
    (0x0008, 0x1200): "DateOfLastCalibration",
    (0x0008, 0x1201): "TimeOfLastCalibration",
    (0x0008, 0x1240): "DimensionOrganizationUID",
    (0x0008, 0x2111): "DerivationDescription",
    (0x0008, 0x2112): "SourceImageSequence",
    (0x0008, 0x2114): "SourceSeriesSequence",
    (0x0008, 0x2115): "SourceStudySequence",
    (0x0008, 0x2116): "SourcePatientSequence",
    (0x0008, 0x2118): "ContentQualification",
    (0x0008, 0x2140): "CustomCharacteristicsSequence",
    (0x0008, 0x2142): "DeviceLabel",
    (0x0008, 0x2143): "DeviceDescription",
    (0x0008, 0x2144): "DeviceLongDescription",
    (0x0008, 0x2145): "DeviceManufacturer",
    (0x0008, 0x2146): "DeviceModelVersion",
    (0x0008, 0x2147): "DeviceSoftwareVersion",
    (0x0008, 0x2148): "ProductionUDI",
    (0x0008, 0x2149): "InventoryUDI",
    (0x0008, 0x214A): "LotNumber",
    (0x0008, 0x214B): "SerialNumber",
    (0x0008, 0x214C): "ExpirationDate",
    (0x0008, 0x214D): "ManufactureDate",
    (0x0008, 0x214E): "DeviceOperator",
    (0x0008, 0x214F): "DeviceIssueDate",
    (0x0008, 0x2150): "UDIProductionIdentifier",
    (0x0008, 0x2151): "StaticCatalogURI",
    (0x0008, 0x2152): "AdditionalInformation",
    (0x0008, 0x2153): "CatalogIdentifier",
    (0x0008, 0x2154): "CatalogVersion",
    (0x0008, 0x2155): "CatalogRevision",
    (0x0008, 0x2156): "CatalogPublicationDate",
    (0x0008, 0x2157): "CatalogExpirationDate",
    (0x0008, 0x2158): "CatalogLanguage",
    (0x0008, 0x2159): "CatalogContactPerson",
    (0x0008, 0x215A): "CatalogContactEmail",
    (0x0008, 0x215B): "CatalogContactPhone",
    (0x0008, 0x215C): "CatalogContactAddress",
    (0x0008, 0x215D): "CatalogContactURL",
    (0x0008, 0x215E): "CatalogContactFax",
    (0x0008, 0x215F): "CatalogContactMobile",
    (0x0008, 0x2160): "CatalogContactTitle",
    (0x0008, 0x2161): "CatalogContactDepartment",
    (0x0008, 0x2162): "CatalogContactOrganization",
    (0x0008, 0x2163): "CatalogContactCountry",
    (0x0008, 0x2164): "CatalogContactStateProvince",
    (0x0008, 0x2165): "CatalogContactCity",
    (0x0008, 0x2166): "CatalogContactPostalCode",
    (0x0008, 0x2167): "CatalogContactStreetAddress",
    (0x0008, 0x2168): "CatalogContactBuilding",
    (0x0008, 0x2169): "CatalogContactRoom",
    (0x0008, 0x216A): "CatalogContactFloor",
    (0x0008, 0x216B): "CatalogContactWing",
    (0x0008, 0x216C): "CatalogContactSection",
    (0x0008, 0x216D): "CatalogContactUnit",
    (0x0008, 0x216E): "CatalogContactSuite",
    (0x0008, 0x216F): "CatalogContactPOBox",
    # Series
    (0x0020, 0x000E): "SeriesInstanceUID",  # Keep for integrity
    (0x0020, 0x0011): "SeriesNumber",
    (0x0020, 0x0012): "AcquisitionNumber",
    (0x0020, 0x0013): "InstanceNumber",
    (0x0020, 0x0014): "IsotopeNumber",
    (0x0020, 0x0015): "PhaseNumber",
    (0x0020, 0x0016): "IntervalNumber",
    (0x0020, 0x0017): "TimeSlotNumber",
    (0x0020, 0x0018): "AngleNumber",
    (0x0020, 0x0019): "TimeSliceNumber",
    (0x0020, 0x0020): "CardiacCyclePosition",
    (0x0020, 0x0030): "ImagePosition",
    (0x0020, 0x0032): "ImagePositionPatient",
    (0x0020, 0x0035): "ImageOrientation",
    (0x0020, 0x0037): "ImageOrientationPatient",
    (0x0020, 0x0050): "SliceLocation",
    (0x0020, 0x0052): "FrameOfReferenceUID",
    (0x0020, 0x0060): "Laterality",
    (0x0020, 0x0062): "ImageLaterality",
    (0x0020, 0x0070): "GeometryOfKSpaceTraversal",
    (0x0020, 0x0071): "SegmentedKSpaceTraversal",
    (0x0020, 0x0072): "RectilinearPhaseEncodeReordering",
    (0x0020, 0x0073): "KSpaceFiltering",
    (0x0020, 0x0074): "TimeDomainFiltering",
    (0x0020, 0x0075): "NumberOfZeroFillsUsed",
    (0x0020, 0x0076): "BaselineCorrection",
    (0x0020, 0x0077): "SegmentNumber",
    (0x0020, 0x0078): "TotalSegmentsRequested",
    (0x0020, 0x0079): "InterpolationSequence",
    (0x0020, 0x0080): "InterpolationNumber",
    (0x0020, 0x0081): "AbsDiffScanAcquisitionTime",
    (0x0020, 0x0090): "NumberOfFrames",
    (0x0020, 0x0091): "FrameIncrementPointer",
    (0x0020, 0x0092): "FrameDimensionPointer",
    (0x0020, 0x0093): "TimingIndexUndefined",
    (0x0020, 0x0094): "TemporalPositionIdentifier",
    (0x0020, 0x0095): "NumberOfTemporalPositions",
    (0x0020, 0x0096): "TemporalResolution",
    (0x0020, 0x0097): "TemporalPositionRangeStart",
    (0x0020, 0x0098): "TemporalPositionRangeEnd",
    (0x0020, 0x0099): "TriggerDelay",
    (0x0020, 0x0100): "TriggerTime",
    (0x0020, 0x0105): "NominalScanningDirection",
    (0x0020, 0x0110): "CardiacBeatRejection",
    (0x0020, 0x0111): "RespiratoryMotionCompensation",
    (0x0020, 0x0112): "ArrhythmiaDetection",
    (0x0020, 0x0113): "VariableFlipAngleFlag",
    (0x0020, 0x0114): "VR_Detection",
    (0x0020, 0x0115): "MaximumHeartRate",
    (0x0020, 0x0116): "MinimumHeartRate",
    (0x0020, 0x0117): "AverageHeartRate",
    (0x0020, 0x0118): "TargetHeartRate",
    (0x0020, 0x0119): "HeartRateVariance",
    (0x0020, 0x0120): "HeartRateMinimumThreshold",
    (0x0020, 0x0121): "HeartRateMaximumThreshold",
    (0x0020, 0x0122): "RRIntervalMean",
    (0x0020, 0x0123): "RRIntervalStandardDeviation",
    (0x0020, 0x0124): "RRIntervalMinimum",
    (0x0020, 0x0125): "RRIntervalMaximum",
    (0x0020, 0x0126): "CardiacSynchronization",
    (0x0020, 0x0127): "CardiacSignalSource",
    (0x0020, 0x0128): "CardiacSignalChannel",
    (0x0020, 0x0129): "CardiacSignalSamplingFrequency",
    (0x0020, 0x0130): "CardiacSignalGain",
    (0x0020, 0x0131): "CardiacSignalOffset",
    (0x0020, 0x0132): "CardiacSignalFilter",
    (0x0020, 0x0133): "CardiacSignalNotchFilter",
    (0x0020, 0x0134): "CardiacSignalBaselineWander",
    (0x0020, 0x0135): "CardiacSignalEMG",
    (0x0020, 0x0136): "CardiacSignalArtifact",
    (0x0020, 0x0137): "CardiacSignalQuality",
    (0x0020, 0x0138): "CardiacSignalLead",
    (0x0020, 0x0139): "CardiacSignalElectrode",
    (0x0020, 0x0140): "CardiacSignalPlacement",
    # Equipment
    (0x0018, 0x0010): "ContrastBolusAgent",
    (0x0018, 0x0012): "ContrastBolusAgentRoute",
    (0x0018, 0x0014): "ContrastBolusAdministrationVolumeSequence",
    (0x0018, 0x0015): "ContrastBolusAdministrationVolumeUnits",
    (0x0018, 0x0016): "ContrastBolusAdministrationVolume",
    (0x0018, 0x0017): "ContrastBolusAgentConcentration",
    (0x0018, 0x0018): "ContrastBolusAgentConcentrationUnits",
    (0x0018, 0x0019): "ContrastBolusAgentConcentrationValue",
    (0x0018, 0x001A): "ContrastBolusAgentSequence",
    (0x0018, 0x0020): "ScanningSequence",
    (0x0018, 0x0021): "SequenceVariant",
    (0x0018, 0x0022): "ScanOptions",
    (0x0018, 0x0023): "MRACquisitionType",
    (0x0018, 0x0024): "SequenceName",
    (0x0018, 0x0025): "AngioFlag",
    (0x0018, 0x0026): "InterventionDrugInformationSequence",
    (0x0018, 0x0027): "InterventionDrugStopTime",
    (0x0018, 0x0028): "InterventionDrugDose",
    (0x0018, 0x0029): "InterventionDrugCodeSequence",
    (0x0018, 0x002A): "AdditionalDrugSequence",
    (0x0018, 0x0030): "Radionuclide",
    (0x0018, 0x0031): "Radiopharmaceutical",
    (0x0018, 0x0032): "EnergyWindowCenterline",
    (0x0018, 0x0033): "EnergyWindowTotalWidth",
    (0x0018, 0x0034): "InterventionDrugName",
    (0x0018, 0x0035): "InterventionDrugStartTime",
    (0x0018, 0x0036): "InterventionSequence",
    (0x0018, 0x0037): "TherapyType",
    (0x0018, 0x0038): "InterventionStatus",
    (0x0018, 0x0039): "TherapyDescription",
    (0x0018, 0x003A): "InterventionDescription",
    (0x0018, 0x0040): "CineRate",
    (0x0018, 0x0042): "InitialCineRunState",
    (0x0018, 0x0050): "SliceThickness",
    (0x0018, 0x0051): "SliceSpacing",
    (0x0018, 0x0052): "SliceProgressionDirection",
    (0x0018, 0x0053): "ScanArc",
    (0x0018, 0x0054): "AngularStep",
    (0x0018, 0x0055): "NumberOfExposures",
    (0x0018, 0x0056): "ReconstructionTargetCenterPatient",
    (0x0018, 0x0057): "SpatialFilterPreservation",
    (0x0018, 0x0058): "ContourUncertaintyRadius",
    (0x0018, 0x0059): "AttachedContourUncertaintyRadiusSequence",
    (0x0018, 0x0060): "DirectDiagonalExposure",
    (0x0018, 0x0061): "AlternateDiagonalExposure",
    (0x0018, 0x0062): "LongitudinalChromaticAberrationCorrection",
    (0x0018, 0x0063): "LateralChromaticAberrationCorrection",
    (0x0018, 0x0064): "MonochromaticAberrationCorrection",
    (0x0018, 0x0065): "DistortionCorrection",
    (0x0018, 0x0066): "NoiseReduction",
    (0x0018, 0x0067): "DynamicRange",
    (0x0018, 0x0068): "ColorSpace",
    (0x0018, 0x0069): "ColorProfile",
    (0x0018, 0x006A): "GammaCurve",
    (0x0018, 0x006B): "ToneCurve",
    (0x0018, 0x006C): "LookUpTable",
    (0x0018, 0x006D): "PaletteColorLookupTableUID",
    (0x0018, 0x006E): "PaletteColorLookupTableSequence",
    (0x0018, 0x006F): "PaletteColorLookupTableDescriptor",
    (0x0018, 0x0070): "PaletteColorLookupTableData",
    (0x0018, 0x0071): "SegmentedPaletteColorLookupTableData",
    (0x0018, 0x0072): "SupplementalPaletteColorLookupTableData",
    (0x0018, 0x0073): "SegmentedPaletteColorLookupTableUID",
    (0x0018, 0x0074): "SupplementalPaletteColorLookupTableUID",
    (0x0018, 0x0075): "DefaultPixelValue",
    (0x0018, 0x0076): "VOILUTFunction",
    (0x0018, 0x0077): "PixelIntensityRelationship",
    (0x0018, 0x0078): "PixelIntensityRelationshipSign",
    (0x0018, 0x0079): "WindowCenter",
    (0x0018, 0x0080): "WindowWidth",
    (0x0018, 0x0081): "RescaleIntercept",
    (0x0018, 0x0082): "RescaleSlope",
    (0x0018, 0x0083): "RescaleType",
    (0x0018, 0x0084): "WindowCenterWidthExplanation",
    (0x0018, 0x0085): "VOILUTFunctionExplanation",
    (0x0018, 0x0086): "PixelIntensityRelationshipExplanation",
    (0x0018, 0x0087): "WindowCenterWidthExplanationExplanation",
    (0x0018, 0x0088): "VOILUTFunctionExplanationExplanation",
    (0x0018, 0x0089): "PixelIntensityRelationshipExplanationExplanation",
    (0x0018, 0x008A): "WindowCenterWidthExplanationExplanationExplanation",
    (0x0018, 0x008B): "VOILUTFunctionExplanationExplanationExplanation",
    (0x0018, 0x008C): "PixelIntensityRelationshipExplanationExplanationExplanation",
    (0x0018, 0x008D): "WindowCenterWidthExplanationExplanationExplanationExplanation",
    (0x0018, 0x008E): "VOILUTFunctionExplanationExplanationExplanationExplanation",
    (0x0018, 0x008F): "PixelIntensityRelationshipExplanationExplanationExplanationExplanation",
    (0x0018, 0x0090): "DataCollectionCenterPatient",
    (0x0018, 0x0091): "RotationalDirection",
    (0x0018, 0x0092): "AngularPosition",
    (0x0018, 0x0093): "RadialPosition",
    (0x0018, 0x0094): "ScanArc",
    (0x0018, 0x0095): "AngularStep",
    (0x0018, 0x0096): "NumberOfExposures",
    (0x0018, 0x0097): "ReconstructionTargetCenterPatient",
    (0x0018, 0x0098): "SpatialFilterPreservation",
    (0x0018, 0x0099): "ContourUncertaintyRadius",
    (0x0018, 0x0100): "AttachedContourUncertaintyRadiusSequence",
    (0x0018, 0x0101): "DirectDiagonalExposure",
    (0x0018, 0x0102): "AlternateDiagonalExposure",
}

# Полная анонимизация (включая дополнительные теги)
FULL_ANONYMIZATION_TAGS = {
    **BASIC_ANONYMIZATION_TAGS,
    # Дополнительные теги для полной анонимизации
    (0x0008, 0x0014): "InstanceCreatorUID",
    (0x0008, 0x0018): "SOPInstanceUID",
    (0x0008, 0x0020): "StudyDate",
    (0x0008, 0x0021): "SeriesDate",
    (0x0008, 0x0022): "AcquisitionDate",
    (0x0008, 0x0023): "ContentDate",
    (0x0008, 0x0024): "OverlayDate",
    (0x0008, 0x0025): "CurveDate",
    (0x0008, 0x002A): "AcquisitionDateTime",
    (0x0008, 0x0030): "StudyTime",
    (0x0008, 0x0031): "SeriesTime",
    (0x0008, 0x0032): "AcquisitionTime",
    (0x0008, 0x0033): "ContentTime",
    (0x0008, 0x0034): "OverlayTime",
    (0x0008, 0x0035): "CurveTime",
    (0x0008, 0x0040): "DataType",
    (0x0008, 0x0041): "DataSubtype",
    (0x0008, 0x0042): "ComputerSystem",
    (0x0008, 0x0043): "LocalizingCursor",
    (0x0008, 0x0044): "BurnedInAnnotation",
    (0x0008, 0x0045): "RecognizableVisualFeatures",
    (0x0008, 0x0046): "LongitudinalTemporalOffsetFromEvent",
    (0x0008, 0x0047): "LongitudinalTemporalEventType",
    (0x0008, 0x0048): "PersonalSubjectCodeSequence",
    (0x0008, 0x0049): "PersonalSubjectCodeValue",
    (0x0008, 0x004A): "PersonalSubjectCodingSchemeDesignator",
    (0x0008, 0x004B): "PersonalSubjectCodingSchemeVersion",
    (0x0008, 0x004C): "PersonalSubjectCodeMeaning",
    (0x0008, 0x004D): "PersonalSubjectTypeOfCode",
    (0x0008, 0x004E): "PersonalSubjectContextSequence",
    (0x0008, 0x004F): "PersonalSubjectContextDescription",
    (0x0008, 0x0051): "SourceProblemID",
    (0x0008, 0x0052): "SourceProblemDescription",
    (0x0008, 0x0053): "SourceProblemCodingScheme",
    (0x0008, 0x0054): "RetrieveAETitle",
    (0x0008, 0x0055): "StationAETitle",
    (0x0008, 0x0056): "InstanceAvailability",
    (0x0008, 0x0058): "FailedSOPInstanceUIDList",
    (0x0008, 0x0059): "WarningReason",
    (0x0008, 0x005A): "FailureReason",
    (0x0008, 0x005B): "ErrorComment",
    (0x0008, 0x005C): "FailureAttributes",
    (0x0008, 0x005D): "FailureAttributeValues",
    (0x0008, 0x005E): "FailureAttributeComments",
    (0x0008, 0x005F): "FailureAttributeCodingScheme",
    (0x0008, 0x0060): "Modality",
    (0x0008, 0x0061): "ModalitiesInStudy",
    (0x0008, 0x0062): "SOPClassesInStudy",
    (0x0008, 0x0063): "AnatomicRegionsInStudy",
    (0x0008, 0x0064): "AnatomicRegionSequence",
    (0x0008, 0x0065): "AnatomicRegionModifierSequence",
    (0x0008, 0x0066): "PrimaryAnatomicStructureSequence",
    (0x0008, 0x0067): "PrimaryAnatomicStructureModifierSequence",
    (0x0008, 0x0068): "TransducerPositionGeographicLocation",
    (0x0008, 0x0069): "TransducerOrientationGeographicLocation",
    (0x0008, 0x006A): "AnatomicStructureSpaceOrdering",
    (0x0008, 0x006B): "AnatomicStructureSpaceOrderingModifier",
    (0x0008, 0x006C): "AnatomicStructureSpaceOrigin",
    (0x0008, 0x006D): "AnatomicStructureSpaceOriginModifier",
    (0x0008, 0x006E): "AnatomicStructureSpaceOriginSequence",
    (0x0008, 0x006F): "AnatomicStructureSpaceOriginModifierSequence",
    (0x0008, 0x0070): "AnatomicStructureSpaceOriginGeographicLocation",
    (0x0008, 0x0071): "AnatomicStructureSpaceOriginGeographicLocationModifier",
    (0x0008, 0x0072): "AnatomicStructureSpaceOriginGeographicLocationSequence",
    (0x0008, 0x0073): "AnatomicStructureSpaceOriginGeographicLocationModifierSequence",
    (0x0008, 0x0074): "AnatomicStructureSpaceOriginGeographicLocationOrigin",
    (0x0008, 0x0075): "AnatomicStructureSpaceOriginGeographicLocationOriginModifier",
    (0x0008, 0x0076): "AnatomicStructureSpaceOriginGeographicLocationOriginSequence",
    (0x0008, 0x0077): "AnatomicStructureSpaceOriginGeographicLocationOriginModifierSequence",
    (0x0008, 0x0078): "AnatomicStructureSpaceOriginGeographicLocationOriginOrigin",
    (0x0008, 0x0079): "AnatomicStructureSpaceOriginGeographicLocationOriginOriginModifier",
    (0x0008, 0x007A): "AnatomicStructureSpaceOriginGeographicLocationOriginOriginSequence",
    (0x0008, 0x007B): "AnatomicStructureSpaceOriginGeographicLocationOriginOriginModifierSequence",
    (0x0008, 0x007C): "AnatomicStructureSpaceOriginGeographicLocationOriginOriginOrigin",
    (0x0008, 0x007D): "AnatomicStructureSpaceOriginGeographicLocationOriginOriginOriginModifier",
    (0x0008, 0x007E): "AnatomicStructureSpaceOriginGeographicLocationOriginOriginOriginSequence",
    (0x0008, 0x007F): "AnatomicStructureSpaceOriginGeographicLocationOriginOriginOriginModifierSequence",
    (0x0008, 0x0080): "InstitutionName",
    (0x0008, 0x0081): "InstitutionAddress",
    (0x0008, 0x0082): "InstitutionCodeSequence",
    (0x0008, 0x0083): "InstitutionDepartmentName",
    (0x0008, 0x0084): "InstitutionDepartmentType",
    (0x0008, 0x0085): "InstitutionDepartmentCodeSequence",
    (0x0008, 0x0086): "InstitutionDepartmentAddress",
    (0x0008, 0x0087): "InstitutionDepartmentPhoneNumber",
    (0x0008, 0x0088): "InstitutionDepartmentEmail",
    (0x0008, 0x0089): "InstitutionDepartmentURL",
    (0x0008, 0x008A): "InstitutionDepartmentFax",
    (0x0008, 0x008B): "InstitutionDepartmentMobile",
    (0x0008, 0x008C): "InstitutionDepartmentContact",
    (0x0008, 0x008D): "InstitutionDepartmentContactTitle",
    (0x0008, 0x008E): "InstitutionDepartmentContactDepartment",
    (0x0008, 0x008F): "InstitutionDepartmentContactOrganization",
    (0x0008, 0x0090): "ReferringPhysicianName",
    (0x0008, 0x0091): "ReferringPhysicianIdentificationSequence",
    (0x0008, 0x0092): "ReferringPhysicianAddress",
    (0x0008, 0x0093): "ReferringPhysicianTelecomInformation",
    (0x0008, 0x0094): "ReferringPhysicianTelephoneNumbers",
    (0x0008, 0x0095): "ReferringPhysicianAddressTelecomInformation",
    (0x0008, 0x0096): "ReferringPhysicianIdentificationSequence",
    (0x0008, 0x0097): "ReferringPhysicianName",
    (0x0008, 0x0098): "ReferringPhysicianIdentificationSequence",
    (0x0008, 0x0099): "ReferringPhysicianAddress",
    (0x0008, 0x009A): "ReferringPhysicianTelecomInformation",
    (0x0008, 0x009B): "ReferringPhysicianTelephoneNumbers",
    (0x0008, 0x009C): "ConsultingPhysicianName",
    (0x0008, 0x009D): "ConsultingPhysicianIdentificationSequence",
    (0x0008, 0x009E): "ConsultingPhysicianAddress",
    (0x0008, 0x009F): "ConsultingPhysicianTelecomInformation",
}


@dataclass
class AnonymizationStats:
    """Статистика анонимизации."""

    total_files: int = 0
    processed_files: int = 0
    failed_files: int = 0
    tags_modified: dict[str, int] = field(default_factory=dict)
    original_values: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnonymizationConfig:
    """Конфигурация анонимизации."""

    level: str = "basic"  # basic, full, research
    preserve_study_integrity: bool = True
    create_mapping_file: bool = True
    output_dir: str | None = None
    dry_run: bool = False
    exclude_tags: set[tuple[int, int]] = field(default_factory=set)
    custom_replacements: dict[tuple[int, int], Any] = field(default_factory=dict)


def generate_pseudo_id(original_value: str, salt: str = "") -> str:
    """Генерирует псевдоанонимный ID на основе хеша."""
    if not original_value:
        return ""
    hash_input = f"{original_value}{salt}"
    return hashlib.sha256(hash_input.encode()).hexdigest()[:16].upper()


def generate_anonymized_date(original_date: str) -> str:
    """Анонимизирует дату, сохраняя относительные различия."""
    if not original_date or len(original_date) < 8:
        return original_date
    try:
        year = original_date[:4]
        month = original_date[4:6]
        day = original_date[6:8]
        # Сдвигаем на фиксированный год для сохранения относительных различий
        base_year = 2000
        year_offset = int(year) - base_year
        anonymized_year = base_year + (year_offset % 100)
        return f"{anonymized_year:04d}{month}{day}"
    except (ValueError, IndexError):
        return original_date


def anonymize_dataset(
    ds: Dataset,
    config: AnonymizationConfig,
    mapping: dict[str, Any],
    stats: AnonymizationStats,
) -> Dataset:
    """
    Анонимизирует DICOM dataset.

    Args:
        ds: DICOM dataset для анонимизации
        config: Конфигурация анонимизации
        mapping: Словарь маппинга оригинальных значений в анонимизированные
        stats: Статистика анонимизации

    Returns:
        Анонимизированный dataset
    """
    if not HAS_PYDICOM:
        raise RuntimeError("Требуется pydicom для анонимизации")

    # Выбор набора тегов в зависимости от уровня
    if config.level == "full":
        tags_to_anonymize = FULL_ANONYMIZATION_TAGS
    else:
        tags_to_anonymize = BASIC_ANONYMIZATION_TAGS

    patient_id = str(ds.get((0x0010, 0x0020), ""))

    for tag, tag_name in tags_to_anonymize.items():
        if tag in config.exclude_tags:
            continue

        if tag not in ds:
            continue

        # Сохраняем UID для целостности исследований
        if config.preserve_study_integrity and tag in [
            (0x0020, 0x000D),  # StudyInstanceUID
            (0x0020, 0x000E),  # SeriesInstanceUID
            (0x0008, 0x0018),  # SOPInstanceUID - можно сохранить или заменить
        ]:
            continue

        original_value = ds[tag].value
        if original_value is None or str(original_value).strip() == "":
            continue

        # Генерация или получение анонимизированного значения
        if tag == (0x0010, 0x0010):  # PatientName
            key = f"patient_name:{patient_id}"
            if key not in mapping:
                mapping[key] = f"ANON_{generate_pseudo_id(str(original_value))}"
            ds[tag].value = mapping[key]
            stats.tags_modified[tag_name] = stats.tags_modified.get(tag_name, 0) + 1

        elif tag == (0x0010, 0x0020):  # PatientID
            key = f"patient_id:{str(original_value)}"
            if key not in mapping:
                mapping[key] = f"PID_{generate_pseudo_id(str(original_value))}"
            ds[tag].value = mapping[key]
            stats.tags_modified[tag_name] = stats.tags_modified.get(tag_name, 0) + 1

        elif tag in [(0x0008, 0x0020), (0x0008, 0x0021), (0x0008, 0x0022)]:  # Dates
            ds[tag].value = generate_anonymized_date(str(original_value))
            stats.tags_modified[tag_name] = stats.tags_modified.get(tag_name, 0) + 1

        elif tag in config.custom_replacements:
            ds[tag].value = config.custom_replacements[tag]
            stats.tags_modified[tag_name] = stats.tags_modified.get(tag_name, 0) + 1

        else:
            # Для остальных текстовых полей
            if isinstance(original_value, str):
                key = f"{tag_name}:{str(original_value)}"
                if key not in mapping:
                    mapping[key] = f"ANON_{generate_pseudo_id(str(original_value))}"
                ds[tag].value = mapping[key]
                stats.tags_modified[tag_name] = stats.tags_modified.get(tag_name, 0) + 1
            elif isinstance(original_value, (int, float)):
                ds[tag].value = 0
                stats.tags_modified[tag_name] = stats.tags_modified.get(tag_name, 0) + 1

    return ds


def anonymize_study(
    study_path: str | Path,
    output_path: str | Path,
    config: AnonymizationConfig,
    mapping: dict[str, Any],
    stats: AnonymizationStats,
) -> tuple[bool, str]:
    """
    Анонимизирует все DICOM файлы в исследовании.

    Args:
        study_path: Путь к исследованию
        output_path: Путь для сохранения анонимизированных данных
        config: Конфигурация анонимизации
        mapping: Словарь маппинга
        stats: Статистика

    Returns:
        Кортеж (успех, сообщение)
    """
    study_path = Path(study_path)
    output_path = Path(output_path)

    if not study_path.exists():
        return False, f"Исследование не найдено: {study_path}"

    # Создание выходной директории
    output_path.mkdir(parents=True, exist_ok=True)

    dicom_files = list(study_path.rglob("*.dcm")) + list(study_path.rglob("*.[dD][iI][cC][oO][mM]"))
    # Также ищем файлы без расширения
    for f in study_path.rglob("*"):
        if f.is_file() and not f.suffix:
            try:
                if pydicom.dcmread(str(f), stop_before_pixels=True):
                    dicom_files.append(f)
            except Exception:
                pass

    if not dicom_files:
        return False, f"DICOM файлы не найдены в {study_path}"

    for dicom_file in dicom_files:
        stats.total_files += 1

        if config.dry_run:
            stats.processed_files += 1
            continue

        try:
            # Чтение файла
            ds = pydicom.dcmread(str(dicom_file), stop_before_pixels=False)

            # Анонимизация
            anon_ds = anonymize_dataset(ds, config, mapping, stats)

            # Определение относительного пути
            rel_path = dicom_file.relative_to(study_path)
            output_file = output_path / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Сохранение
            anon_ds.save_as(str(output_file), write_like_original=True)
            stats.processed_files += 1

            logger.debug("Анонимизирован файл: %s", dicom_file)

        except Exception as e:
            stats.failed_files += 1
            error_msg = f"Ошибка обработки {dicom_file}: {e!s}"
            logger.error(error_msg)
            return False, error_msg

    # Сохранение mapping файла если нужно
    if config.create_mapping_file and not config.dry_run:
        mapping_file = output_path / "anonymization_mapping.json"
        with open(mapping_file, "w", encoding="utf-8") as f:  # type: ignore[assignment, arg-type]
            json.dump(mapping, f, indent=2, ensure_ascii=False)  # type: ignore[arg-type]
        logger.info("Файл маппинга сохранён: %s", mapping_file)

    return True, f"Успешно анонимизировано {stats.processed_files} файлов"


def run_anonymization(
    input_path: str | Path,
    output_path: str | Path,
    config: AnonymizationConfig,
) -> tuple[AnonymizationStats, dict[str, Any]]:
    """
    Запускает анонимизацию всего датасета.

    Args:
        input_path: Путь к исходному датасету
        output_path: Путь для анонимизированных данных
        config: Конфигурация

    Returns:
        Кортеж (статистика, общий маппинг)
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    stats = AnonymizationStats()
    global_mapping: dict[str, Any] = {}

    # Поиск всех исследований
    studies = []
    for item in input_path.iterdir():
        if item.is_dir():
            studies.append(item)

    if not studies:
        # Возможно, это одно исследование
        studies = [input_path]

    for study in studies:
        study_output = output_path / study.name
        success, msg = anonymize_study(study, study_output, config, global_mapping, stats)
        if not success:
            logger.warning("Не удалось анонимизировать %s: %s", study, msg)

    return stats, global_mapping

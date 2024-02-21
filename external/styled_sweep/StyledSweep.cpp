//Sweep

// Mandatory UF Includes
#include <uf.h>
#include <uf_curve.h>
#include <uf_disp.h>
#include <uf_defs.h>
#include <uf_modl.h>
#include <uf_obj.h>
#include <uf_object_types.h>
#include <uf_part.h>
#include <uf_ui.h>
#include <uf_vec.h>

// Internal+External Includes
#include <NXOpen/Annotations.hxx>
#include <NXOpen/Assemblies_Component.hxx>
#include <NXOpen/Assemblies_ComponentAssembly.hxx>
#include <NXOpen/Arc.hxx>
#include <NXOpen/ArcCollection.hxx>
#include <NXOpen/Body.hxx>
#include <NXOpen/BodyCollection.hxx>
#include <NXOpen/BasePart.hxx>
#include <NXOpen/Builder.hxx>
#include <NXOpen/BaseCreator.hxx>
#include <NXOpen/BaseImporter.hxx>
#include <NXOpen/BodyFeatureRule.hxx>
#include <NXOpen/CartesianCoordinateSystem.hxx>
#include <NXOpen/CoordinateSystemCollection.hxx>
#include <NXOpen/CurveDumbRule.hxx>
#include <NXOpen/CurveFeatureRule.hxx>
#include <NXOpen/DexBuilder.hxx>
#include <NXOpen/DexManager.hxx>
#include <NXOpen/Direction.hxx>
#include <NXOpen/DisplayManager.hxx>
#include <NXOpen/DisplayModification.hxx>
#include <NXOpen/DisplayableObject.hxx>
#include <NXOpen/Expression.hxx>
#include <NXOpen/ExpressionCollection.hxx>
#include <NXOpen/Features_DatumCsys.hxx>
#include <NXOpen/Face.hxx>
#include <NXOpen/Features_ExtractFaceBuilder.hxx>
#include <NXOpen/Features_Feature.hxx>
#include <NXOpen/Features_FeatureCollection.hxx>
#include <NXOpen/Features_FitCurve.hxx>
#include <NXOpen/Features_PointSet.hxx>
#include <NXOpen/Features_PointSetBuilder.hxx>
#include <NXOpen/Features_PointSetFacePercentageBuilder.hxx>
#include <NXOpen/Features_PointSetFacePercentageBuilderList.hxx>
#include <NXOpen/Features_StyledSweep.hxx>
#include <NXOpen/Features_StyledSweepBuilder.hxx>
#include <NXOpen/Features_FitCurveBuilder.hxx>
#include <NXOpen/Features_GeometricConstraintData.hxx>
#include <NXOpen/Features_GeometricConstraintDataManager.hxx>
#include <NXOpen/Features_HelixBuilder.hxx>
#include <NXOpen/Features_CompositeCurve.hxx>
#include <NXOpen/FileNew.hxx>
#include <NXOpen/GeometricUtilities_CurveExtensionBuilder.hxx>
#include <NXOpen/GeometricUtilities_PointsFromFileBuilder.hxx>
#include <NXOpen/GeometricUtilities_EntityUsageInfo.hxx>
#include <NXOpen/GeometricUtilities_EntityUsageInfoList.hxx>
#include <NXOpen/GeometricUtilities_OnPathDimWithValueBuilder.hxx>
#include <NXOpen/GeometricUtilities_OnPathDimensionBuilder.hxx>
#include <NXOpen/GeometricUtilities_ParentEquivalencyMap.hxx>
#include <NXOpen/GeometricUtilities_ParentEquivalencyMapList.hxx>
#include <NXOpen/GeometricUtilities_Rebuild.hxx>
#include <NXOpen/GeometricUtilities_ReplAsstBuilder.hxx>
#include <NXOpen/GeometricUtilities_RotationSetBuilder.hxx>
#include <NXOpen/GeometricUtilities_RotationSetBuilderList.hxx>
#include <NXOpen/GeometricUtilities_ScalingSetBuilder.hxx>
#include <NXOpen/GeometricUtilities_ScalingSetBuilderList.hxx>
#include <NXOpen/GeometricUtilities_StyledSweepDoubleOnPathDimBuilder.hxx>
#include <NXOpen/GeometricUtilities_StyledSweepDoubleOnPathDimBuilderList.hxx>
#include <NXOpen/GeometricUtilities_StyledSweepReferenceMethodBuilder.hxx>
#include <NXOpen/GeometricUtilities_SurfaceRangeBuilder.hxx>
#include <NXOpen/Group.hxx>
#include <NXOpen/IBaseCurve.hxx>
#include <NXOpen/Line.hxx>
#include <NXOpen/ListingWindow.hxx>
#include <NXOpen/Layer.hxx>
#include <NXOpen/Layer_LayerManager.hxx>
#include <NXOpen/MeasureManager.hxx>
#include <NXOpen/NXException.hxx>
#include <NXOpen/NXObject.hxx>
#include <NXOpen/NXObjectList.hxx>
#include <NXOpen/NXMessageBox.hxx>
#include <NXOpen/ObjectList.hxx>
#include <NXOpen/ObjectSelector.hxx>
#include <NXOpen/ObjectTypeSelector.hxx>
#include <NXOpen/Part.hxx>
#include <NXOpen/PartCollection.hxx>
#include <NXOpen/Point.hxx>
#include <NXOpen/PointCollection.hxx>
#include <NXOpen/PreviewBuilder.hxx>
#include <NXOpen/SelectSpline.hxx>
#include <NXOpen/SelectTaggedObjectList.hxx>
#include <NXOpen/Scalar.hxx>
#include <NXOpen/ScalarCollection.hxx>
#include <NXOpen/ScCollector.hxx>
#include <NXOpen/ScRuleFactory.hxx>
#include <NXOpen/Section.hxx>
#include <NXOpen/SectionCollection.hxx>
#include <NXOpen/SectionList.hxx>
#include <NXOpen/SelectDisplayableObjectList.hxx>
#include <NXOpen/SelectFace.hxx>
#include <NXOpen/SelectFaceList.hxx>
#include <NXOpen/SelectNXObjectList.hxx>
#include <NXOpen/SelectObject.hxx>
#include <NXOpen/SelectObjectList.hxx>
#include <NXOpen/SelectionIntentRule.hxx>
#include <NXOpen/SelectionIntentRuleOptions.hxx>
#include <NXOpen/Session.hxx>
#include <NXOpen/Spline.hxx>
#include <NXOpen/SplineCollection.hxx>
#include <NXOpen/Step214Importer.hxx>
#include <NXOpen/StepCreator.hxx>
#include <NXOpen/TaggedObject.hxx>
#include <NXOpen/Unit.hxx>
#include <NXOpen/Update.hxx>
#include <NXOpen/UnitCollection.hxx>
#include <NXOpen/UI.hxx>
#include <NXOpen/XformCollection.hxx>

// Std C++ Includes
#include <iostream>
#include <cstdlib>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <cstdio>
#include <exception>
#include <io.h>


using namespace NXOpen;
using std::string;
using std::exception;
using std::stringstream;
using std::ifstream;
using std::vector;
using std::endl;
using std::cout;
using std::cerr;

char* UGII_BASE_DIR = ""; // UG环境变量

static void doCreatePart(Session* theSession, const string PARTNAME) {
    FileNew* fileNew1;

    fileNew1 = theSession->Parts()->FileNew();
    fileNew1->SetTemplateFileName("model-plain-1-mm-template.prt");
    fileNew1->SetUseBlankTemplate(false);
    fileNew1->SetApplicationName("ModelTemplate");
    fileNew1->SetUnits(NXOpen::Part::UnitsMillimeters);
    fileNew1->SetRelationType("");
    fileNew1->SetUsesMasterModel("No");
    fileNew1->SetTemplateType(FileNewTemplateTypeItem);
    fileNew1->SetTemplatePresentationName(NXString("\346\250\241\345\236\213", NXString::UTF8));
    fileNew1->SetItemType("");
    fileNew1->SetSpecialization("");
    fileNew1->SetCanCreateAltrep(false);
    fileNew1->SetNewFileName(PARTNAME);
    fileNew1->SetMasterFileName("");
    fileNew1->SetMakeDisplayedPart(true);
    fileNew1->SetDisplayPartOption(NXOpen::DisplayPartOptionAllowAdditional);
    fileNew1->Commit();
    fileNew1->Destroy();

    theSession->ApplicationSwitchImmediate("UG_APP_MODELING");
}

NXObject* doCurveFit(Part* workPart, Part* desplay, string path, Point3d& helpPoint, int degree = 5) {
    ifstream getPoints;
    getPoints.open(path);

    vector<TaggedObject*> object;
    double x, y, z;
    while (getPoints >> x >> y >> z) {
        Scalar* scalarx, * scalary, * scalarz;
        scalarx = workPart->Scalars()->CreateScalar(x, Scalar::DimensionalityTypeNone, SmartObject::UpdateOptionWithinModeling);
        scalary = workPart->Scalars()->CreateScalar(y, Scalar::DimensionalityTypeNone, SmartObject::UpdateOptionWithinModeling);
        scalarz = workPart->Scalars()->CreateScalar(z, Scalar::DimensionalityTypeNone, SmartObject::UpdateOptionWithinModeling);
        Point* point(dynamic_cast<Point*>(workPart->Points()->CreatePoint(scalarx, scalary, scalarz, SmartObject::UpdateOptionWithinModeling)));
        object.emplace_back(point);
    }
    getPoints.close();
    helpPoint = dynamic_cast<Point*>(object[0])->Coordinates();
    Features::FitCurve* nullNXOpen_Features_FitCurve(NULL);
    Features::FitCurveBuilder* fitCurveBuilder;
    fitCurveBuilder = workPart->Features()->CreateFitCurveBuilder(nullNXOpen_Features_FitCurve);
    fitCurveBuilder->SetTolerance(0.001);
    fitCurveBuilder->SetProjectionDirectionOption(Features::FitCurveBuilder::ProjectionDirectionOptionsNormal);
    fitCurveBuilder->Radius()->SetFormula("50");
    fitCurveBuilder->SetDegree(degree);
    fitCurveBuilder->Extender()->StartValue()->SetFormula("0");
    fitCurveBuilder->Extender()->EndValue()->SetFormula("0");
    fitCurveBuilder->RejectionThreshold()->SetFormula("10");
    fitCurveBuilder->Target()->Add(object);
    fitCurveBuilder->SetHasReversedDirection(true);
    NXObject* nXObject;
    nXObject = fitCurveBuilder->Commit();
    delete nullNXOpen_Features_FitCurve;
    fitCurveBuilder->Destroy();
    return nXObject;
}

vector<NXObject*> getSectionLine(Session* theSession, Part* workPart, string path) {
    string outputfile = std::to_string(rand() % 100000);

    Step214Importer* step214Importer1;

    step214Importer1 = theSession->DexManager()->CreateStep214Importer();
    step214Importer1->SetSimplifyGeometry(true);
    step214Importer1->SetLayerDefault(1);
    step214Importer1->SetOutputFile((outputfile + ".prt").c_str());
    step214Importer1->SetSettingsFile(strcat(UGII_BASE_DIR, "\\step214ug\\step214ug.def"));
    step214Importer1->SetMode(BaseImporter::ModeNativeFileSystem);
    step214Importer1->SetInputFile(path);
    step214Importer1->SetFileOpenFlag(false);
    step214Importer1->SetProcessHoldFlag(true);
    step214Importer1->Commit();
    step214Importer1->Destroy();

    string oscmd = "del " + outputfile + ".log";
    system(oscmd.c_str());
    auto AllObjects = workPart->Layers()->GetAllObjectsOnLayer(1);
    return AllObjects;
}

static void doStyledSweepwithDir(Session* theSession, Part* workPart, Part* display, vector<NXObject*> sections, NXObject* guideCurve, Point3d helpPointGuide, NXObject* dirCurve, Point3d helpPointDir, string savePath, const string PARTNAME){

    NXOpen::Features::Feature* nullNXOpen_Features_Feature(NULL);
    NXOpen::Features::StyledSweepBuilder* styledSweepBuilder1;
    styledSweepBuilder1 = workPart->Features()->CreateStyledSweepBuilder(nullNXOpen_Features_Feature);

    NXOpen::Unit* unit1(dynamic_cast<NXOpen::Unit*>(workPart->UnitCollection()->FindObject("MilliMeter")));
    NXOpen::Expression* expression1;
    expression1 = workPart->Expressions()->CreateSystemExpressionWithUnits("0", unit1);

    NXOpen::Expression* expression2;
    expression2 = workPart->Expressions()->CreateSystemExpressionWithUnits("0", unit1);

    NXOpen::Expression* expression3;
    expression3 = workPart->Expressions()->CreateSystemExpressionWithUnits("0", unit1);

    NXOpen::Expression* expression4;
    expression4 = workPart->Expressions()->CreateSystemExpressionWithUnits("0", unit1);

    styledSweepBuilder1->SurfaceRange()->UStart()->Expression()->SetFormula("0");

    styledSweepBuilder1->SurfaceRange()->UStart()->Expression()->SetFormula("0");

    styledSweepBuilder1->SurfaceRange()->UEnd()->Expression()->SetFormula("100");

    styledSweepBuilder1->SurfaceRange()->UEnd()->Expression()->SetFormula("100");

    styledSweepBuilder1->SurfaceRange()->VStart()->Expression()->SetFormula("0");

    styledSweepBuilder1->SurfaceRange()->VStart()->Expression()->SetFormula("0");

    styledSweepBuilder1->SurfaceRange()->VEnd()->Expression()->SetFormula("100");

    styledSweepBuilder1->SurfaceRange()->VEnd()->Expression()->SetFormula("100");

    styledSweepBuilder1->SetG0Tolerance(0.01);

    styledSweepBuilder1->SetG1Tolerance(0.5);

    styledSweepBuilder1->SetType(NXOpen::Features::StyledSweepBuilder::TypesOneGuideOneOrientation);

    styledSweepBuilder1->SetTransitionOption(NXOpen::Features::StyledSweepBuilder::TransitionOptionsBlend);

    styledSweepBuilder1->SetFixedStringOption(NXOpen::Features::StyledSweepBuilder::FixedStringOptionsGuideAndSection);

    styledSweepBuilder1->SetSectionOrientationOption(NXOpen::Features::StyledSweepBuilder::SectionOrientationOptionsKeepAngle);

    styledSweepBuilder1->SurfaceRange()->SetAnchorPosition(NXOpen::GeometricUtilities::SurfaceRangeBuilder::AnchorPositionTypeVertex1);

    styledSweepBuilder1->SurfaceRange()->UStart()->Expression()->SetFormula("0");

    styledSweepBuilder1->SurfaceRange()->UEnd()->Expression()->SetFormula("100");

    styledSweepBuilder1->SurfaceRange()->VStart()->Expression()->SetFormula("0");

    styledSweepBuilder1->SurfaceRange()->VEnd()->Expression()->SetFormula("100");

    styledSweepBuilder1->SurfaceRange()->VStart()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepBuilder1->SurfaceRange()->VEnd()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepBuilder1->SurfaceRange()->UStart()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepBuilder1->SurfaceRange()->UEnd()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepBuilder1->FirstGuide()->SetDistanceTolerance(0.01);

    styledSweepBuilder1->FirstGuide()->SetChainingTolerance(0.0095);

    styledSweepBuilder1->SecondGuide()->SetDistanceTolerance(0.01);

    styledSweepBuilder1->SecondGuide()->SetChainingTolerance(0.0095);

    styledSweepBuilder1->ReferenceMethod()->ReferenceCurve()->SetDistanceTolerance(0.01);

    styledSweepBuilder1->ReferenceMethod()->ReferenceCurve()->SetChainingTolerance(0.0095);

    styledSweepBuilder1->ScalingCurve()->SetDistanceTolerance(0.01);

    styledSweepBuilder1->ScalingCurve()->SetChainingTolerance(0.0095);

    styledSweepBuilder1->FirstGuide()->SetAngleTolerance(0.5);

    styledSweepBuilder1->SecondGuide()->SetAngleTolerance(0.5);

    styledSweepBuilder1->ReferenceMethod()->ReferenceCurve()->SetAngleTolerance(0.5);

    styledSweepBuilder1->ScalingCurve()->SetAngleTolerance(0.5);

    NXOpen::Section* section1;
    section1 = workPart->Sections()->CreateSection(0.0095, 0.01, 0.5);

    styledSweepBuilder1->SectionList()->Append(section1);

    NXOpen::Curve* nullNXOpen_Curve(NULL);
    NXOpen::GeometricUtilities::StyledSweepDoubleOnPathDimBuilder* styledSweepDoubleOnPathDimBuilder1;
    styledSweepDoubleOnPathDimBuilder1 = styledSweepBuilder1->CreatePivotSet(0.0, 0.0, nullNXOpen_Curve, nullNXOpen_Curve);

    styledSweepDoubleOnPathDimBuilder1->FirstLocation()->Expression()->SetFormula("0");

    styledSweepDoubleOnPathDimBuilder1->FirstLocation()->Expression()->SetFormula("0");

    styledSweepDoubleOnPathDimBuilder1->SecondLocation()->Expression()->SetFormula("0");

    styledSweepDoubleOnPathDimBuilder1->SecondLocation()->Expression()->SetFormula("0");

    styledSweepBuilder1->PivotSetList()->Append(styledSweepDoubleOnPathDimBuilder1);

    NXOpen::GeometricUtilities::RotationSetBuilder* rotationSetBuilder1;
    rotationSetBuilder1 = styledSweepBuilder1->CreateRotationSet(0.0, 100.0, nullNXOpen_Curve);

    rotationSetBuilder1->Location()->Expression()->SetFormula("100");

    rotationSetBuilder1->Location()->Expression()->SetFormula("100");

    styledSweepBuilder1->RotationSetList()->Append(rotationSetBuilder1);

    NXOpen::GeometricUtilities::ScalingSetBuilder* scalingSetBuilder1;
    scalingSetBuilder1 = styledSweepBuilder1->CreateScalingSet(100.0, 100.0, 50.0, nullNXOpen_Curve);

    scalingSetBuilder1->Location()->Expression()->SetFormula("50");

    scalingSetBuilder1->Location()->Expression()->SetFormula("50");

    styledSweepBuilder1->ScalingSetList()->Append(scalingSetBuilder1);

    section1->SetAllowedEntityTypes(NXOpen::Section::AllowTypesOnlyCurves);

    NXOpen::SelectionIntentRuleOptions* selectionIntentRuleOptions1;
    selectionIntentRuleOptions1 = workPart->ScRuleFactory()->CreateRuleOptions();

    selectionIntentRuleOptions1->SetSelectedFromInactive(false);

    vector<IBaseCurve* >curves1;
    for (auto& se : sections) curves1.emplace_back(dynamic_cast<IBaseCurve*>(se));

    NXOpen::CurveDumbRule* curveDumbRule1;
    curveDumbRule1 = workPart->ScRuleFactory()->CreateRuleBaseCurveDumb(curves1, selectionIntentRuleOptions1);

    delete selectionIntentRuleOptions1;
    section1->AllowSelfIntersection(true);

    section1->AllowDegenerateCurves(false);

    std::vector<NXOpen::SelectionIntentRule*> rules1(1);
    rules1[0] = curveDumbRule1;
    NXOpen::NXObject* nullNXOpen_NXObject(NULL);
    NXOpen::Point3d helpPoint1(0.0, 0.0, 0.0);
    section1->AddToSection(rules1, nullNXOpen_NXObject, nullNXOpen_NXObject, nullNXOpen_NXObject, helpPoint1, NXOpen::Section::ModeCreate, false);

    // section1->ReverseDirectionOfLoop(0);

    styledSweepBuilder1->FirstGuide()->SetAllowedEntityTypes(NXOpen::Section::AllowTypesOnlyCurves);

    NXOpen::SelectionIntentRuleOptions* selectionIntentRuleOptions2;
    selectionIntentRuleOptions2 = workPart->ScRuleFactory()->CreateRuleOptions();

    selectionIntentRuleOptions2->SetSelectedFromInactive(false);

    std::vector<NXOpen::Features::Feature*> features1(1);
    NXOpen::Features::FitCurve* fitCurve1(dynamic_cast<NXOpen::Features::FitCurve*>(guideCurve));
    features1[0] = fitCurve1;
    NXOpen::DisplayableObject* nullNXOpen_DisplayableObject(NULL);
    NXOpen::CurveFeatureRule* curveFeatureRule1;
    curveFeatureRule1 = workPart->ScRuleFactory()->CreateRuleCurveFeature(features1, nullNXOpen_DisplayableObject, selectionIntentRuleOptions2);

    delete selectionIntentRuleOptions2;
    styledSweepBuilder1->FirstGuide()->AllowSelfIntersection(true);

    styledSweepBuilder1->FirstGuide()->AllowDegenerateCurves(false);

    std::vector<NXOpen::SelectionIntentRule*> rules2(1);
    rules2[0] = curveFeatureRule1;
    NXOpen::Point3d helpPoint2 = helpPointGuide;
    styledSweepBuilder1->FirstGuide()->AddToSection(rules2, nullNXOpen_NXObject, nullNXOpen_NXObject, nullNXOpen_NXObject, helpPoint2, NXOpen::Section::ModeCreate, false);

    styledSweepBuilder1->SecondGuide()->SetAllowedEntityTypes(NXOpen::Section::AllowTypesOnlyCurves);

    NXOpen::SelectionIntentRuleOptions* selectionIntentRuleOptions3;
    selectionIntentRuleOptions3 = workPart->ScRuleFactory()->CreateRuleOptions();

    selectionIntentRuleOptions3->SetSelectedFromInactive(false);

    std::vector<NXOpen::Features::Feature*> features2(1);
    NXOpen::Features::FitCurve* fitCurve2(dynamic_cast<NXOpen::Features::FitCurve*>(dirCurve));
    features2[0] = fitCurve2;
    NXOpen::CurveFeatureRule* curveFeatureRule2;
    curveFeatureRule2 = workPart->ScRuleFactory()->CreateRuleCurveFeature(features2, nullNXOpen_DisplayableObject, selectionIntentRuleOptions3);

    delete selectionIntentRuleOptions3;
    styledSweepBuilder1->SecondGuide()->AllowSelfIntersection(true);

    styledSweepBuilder1->SecondGuide()->AllowDegenerateCurves(false);

    std::vector<NXOpen::SelectionIntentRule*> rules3(1);
    rules3[0] = curveFeatureRule2;
    NXOpen::Point3d helpPoint3 = helpPointDir;
    styledSweepBuilder1->SecondGuide()->AddToSection(rules3, nullNXOpen_NXObject, nullNXOpen_NXObject, nullNXOpen_NXObject, helpPoint3, NXOpen::Section::ModeCreate, false);

    styledSweepDoubleOnPathDimBuilder1->FirstLocation()->Expression()->SetRightHandSide("0");

    styledSweepDoubleOnPathDimBuilder1->SecondLocation()->Expression()->SetRightHandSide("0");

    rotationSetBuilder1->Value()->SetRightHandSide("0");

    rotationSetBuilder1->Location()->Expression()->SetRightHandSide("100");

    scalingSetBuilder1->ScalingValue()->SetRightHandSide("100");

    scalingSetBuilder1->Value()->SetRightHandSide("100");

    scalingSetBuilder1->Location()->Expression()->SetRightHandSide("50");

    styledSweepBuilder1->SurfaceRange()->UStart()->Expression()->SetRightHandSide("0");

    styledSweepBuilder1->SurfaceRange()->UEnd()->Expression()->SetRightHandSide("100");

    styledSweepBuilder1->SurfaceRange()->VStart()->Expression()->SetRightHandSide("0");

    styledSweepBuilder1->SurfaceRange()->VEnd()->Expression()->SetRightHandSide("100");

    styledSweepDoubleOnPathDimBuilder1->FirstLocation()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepDoubleOnPathDimBuilder1->SecondLocation()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    rotationSetBuilder1->Location()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    scalingSetBuilder1->Location()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepBuilder1->SurfaceRange()->VStart()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepBuilder1->SurfaceRange()->VEnd()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepBuilder1->SurfaceRange()->UStart()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepBuilder1->SurfaceRange()->UEnd()->Update(NXOpen::GeometricUtilities::OnPathDimensionBuilder::UpdateReasonPath);

    styledSweepDoubleOnPathDimBuilder1->FirstLocation()->Expression()->SetRightHandSide("0");

    styledSweepDoubleOnPathDimBuilder1->SecondLocation()->Expression()->SetRightHandSide("0");

    rotationSetBuilder1->Value()->SetRightHandSide("0");

    rotationSetBuilder1->Location()->Expression()->SetRightHandSide("100");

    scalingSetBuilder1->ScalingValue()->SetRightHandSide("100");

    scalingSetBuilder1->Value()->SetRightHandSide("100");

    scalingSetBuilder1->Location()->Expression()->SetRightHandSide("50");

    styledSweepBuilder1->SurfaceRange()->UStart()->Expression()->SetRightHandSide("0");

    styledSweepBuilder1->SurfaceRange()->UEnd()->Expression()->SetRightHandSide("100");

    styledSweepBuilder1->SurfaceRange()->VStart()->Expression()->SetRightHandSide("0");

    styledSweepBuilder1->SurfaceRange()->VEnd()->Expression()->SetRightHandSide("100");

    NXOpen::NXObject* nXObject1;
    nXObject1 = styledSweepBuilder1->Commit();

    styledSweepBuilder1->Destroy();

    workPart->MeasureManager()->SetPartTransientModification();

    workPart->Expressions()->Delete(expression1);

    workPart->MeasureManager()->ClearPartTransientModification();

    workPart->MeasureManager()->SetPartTransientModification();

    workPart->Expressions()->Delete(expression2);

    workPart->MeasureManager()->ClearPartTransientModification();

    workPart->MeasureManager()->SetPartTransientModification();

    workPart->Expressions()->Delete(expression3);

    workPart->MeasureManager()->ClearPartTransientModification();

    workPart->MeasureManager()->SetPartTransientModification();

    workPart->Expressions()->Delete(expression4);

    workPart->MeasureManager()->ClearPartTransientModification();

    theSession->UpdateManager()->SetInterpartDelay(false);

    StepCreator* stepCreator1;
    stepCreator1 = theSession->DexManager()->CreateStepCreator();

    stepCreator1->SetExportAs(StepCreator::ExportAsOptionAp214);

    stepCreator1->SetSettingsFile(strcat(UGII_BASE_DIR, "\\step214ug\\ugstep214.def"));

    stepCreator1->ExportSelectionBlock()->SetSelectionScope(ObjectSelector::ScopeSelectedObjects);

    stepCreator1->SetColorAndLayers(true);

    stepCreator1->SetBsplineTol(0.0001);

    stepCreator1->SetInputFile(PARTNAME);

    Body* body1(dynamic_cast<Body*>(workPart->Bodies()->FindObject("STYLED_SWEEP(3)")));
    bool added1;
    added1 = stepCreator1->ExportSelectionBlock()->SelectionComp()->Add(body1);

    stepCreator1->SetOutputFile(savePath.c_str());

    stepCreator1->SetFileSaveFlag(false);

    stepCreator1->SetLayerMask("1-256");

    stepCreator1->SetProcessHoldFlag(true);

    NXObject* nXObject4;
    nXObject4 = stepCreator1->Commit();

    stepCreator1->Destroy();

}

static void doSampleCurve(Session* theSession, Part* workPart, Part* display, NXObject* Curve, string save_path, int sample_num, const string PARTNAME) {
    NXOpen::Features::PointSet* nullNXOpen_Features_PointSet(NULL);
    NXOpen::Features::PointSetBuilder* pointSetBuilder1;

    pointSetBuilder1 = workPart->Features()->CreatePointSetBuilder(nullNXOpen_Features_PointSet);

    pointSetBuilder1->NumberOfPointsExpression()->SetFormula(std::to_string(sample_num).c_str());

    pointSetBuilder1->StartPercentage()->SetFormula("0");

    pointSetBuilder1->EndPercentage()->SetFormula("100");

    pointSetBuilder1->Ratio()->SetFormula("1");

    pointSetBuilder1->ChordalTolerance()->SetFormula("2.54");

    pointSetBuilder1->ArcLength()->SetFormula("1");

    pointSetBuilder1->NumberOfPointsInUDirectionExpression()->SetFormula("10");

    pointSetBuilder1->NumberOfPointsInVDirectionExpression()->SetFormula("10");

    pointSetBuilder1->SetPatternLimitsBy(NXOpen::Features::PointSetBuilder::PatternLimitsTypePercentages);

    pointSetBuilder1->PatternLimitsStartingUValue()->SetFormula("0");

    pointSetBuilder1->PatternLimitsEndingUValue()->SetFormula("100");

    pointSetBuilder1->PatternLimitsStartingVValue()->SetFormula("0");

    pointSetBuilder1->PatternLimitsEndingVValue()->SetFormula("100");

    NXOpen::Unit* nullNXOpen_Unit(NULL);
    NXOpen::Expression* expression1;
    expression1 = workPart->Expressions()->CreateSystemExpressionWithUnits("50", nullNXOpen_Unit);

    pointSetBuilder1->CurvePercentageList()->Append(expression1);

    NXOpen::Features::PointSetFacePercentageBuilder* pointSetFacePercentageBuilder1;
    pointSetFacePercentageBuilder1 = pointSetBuilder1->CreateFacePercentageListItem();

    pointSetBuilder1->FacePercentageList()->Append(pointSetFacePercentageBuilder1);

    expression1->SetFormula("0");

    pointSetFacePercentageBuilder1->UPercentage()->SetFormula("0");

    pointSetFacePercentageBuilder1->VPercentage()->SetFormula("0");

    pointSetBuilder1->SingleCurveOrEdgeCollector()->SetDistanceTolerance(0.001);

    pointSetBuilder1->SingleCurveOrEdgeCollector()->SetChainingTolerance(0.00095);

    pointSetBuilder1->MultipleCurveOrEdgeCollector()->SetDistanceTolerance(0.001);

    pointSetBuilder1->MultipleCurveOrEdgeCollector()->SetChainingTolerance(0.00095);

    pointSetBuilder1->StartPercentageSection()->SetDistanceTolerance(0.001);

    pointSetBuilder1->StartPercentageSection()->SetChainingTolerance(0.00095);

    pointSetBuilder1->EndPercentageSection()->SetDistanceTolerance(0.001);

    pointSetBuilder1->EndPercentageSection()->SetChainingTolerance(0.00095);

    pointSetBuilder1->SingleCurveOrEdgeCollector()->SetAngleTolerance(0.050000000000000003);

    pointSetBuilder1->MultipleCurveOrEdgeCollector()->SetAngleTolerance(0.050000000000000003);

    pointSetBuilder1->StartPercentageSection()->SetAngleTolerance(0.050000000000000003);

    pointSetBuilder1->EndPercentageSection()->SetAngleTolerance(0.050000000000000003);

    pointSetBuilder1->SingleCurveOrEdgeCollector()->SetAllowedEntityTypes(NXOpen::Section::AllowTypesOnlyCurves);

    NXOpen::SelectionIntentRuleOptions* selectionIntentRuleOptions1;
    selectionIntentRuleOptions1 = workPart->ScRuleFactory()->CreateRuleOptions();

    selectionIntentRuleOptions1->SetSelectedFromInactive(false);

    std::vector<NXOpen::Features::Feature*> features1(1);
    NXOpen::Features::FitCurve* fitCurve1(dynamic_cast<Features::FitCurve*>(Curve));
    features1[0] = fitCurve1;
    NXOpen::DisplayableObject* nullNXOpen_DisplayableObject(NULL);
    NXOpen::CurveFeatureRule* curveFeatureRule1;
    curveFeatureRule1 = workPart->ScRuleFactory()->CreateRuleCurveFeature(features1, nullNXOpen_DisplayableObject, selectionIntentRuleOptions1);

    delete selectionIntentRuleOptions1;
    pointSetBuilder1->SingleCurveOrEdgeCollector()->AllowSelfIntersection(true);

    pointSetBuilder1->SingleCurveOrEdgeCollector()->AllowDegenerateCurves(false);

    std::vector<NXOpen::SelectionIntentRule*> rules1(1);
    rules1[0] = curveFeatureRule1;
    NXOpen::NXObject* nullNXOpen_NXObject(NULL);
    NXOpen::Point3d helpPoint1(0.0, 0.0, 0.0);
    pointSetBuilder1->SingleCurveOrEdgeCollector()->AddToSection(rules1, nullNXOpen_NXObject, nullNXOpen_NXObject, nullNXOpen_NXObject, helpPoint1, NXOpen::Section::ModeCreate, false);
    
    NXOpen::NXObject* nXObject1;
    nXObject1 = pointSetBuilder1->Commit();

    NXOpen::Expression* expression2(pointSetBuilder1->StartPercentage());
    NXOpen::Expression* expression3(pointSetBuilder1->NumberOfPointsExpression());
    NXOpen::Expression* expression4(pointSetBuilder1->EndPercentage());
    pointSetBuilder1->Destroy();
    

    NXOpen::StepCreator* stepCreator1;
    stepCreator1 = theSession->DexManager()->CreateStepCreator();

    stepCreator1->SetExportAs(NXOpen::StepCreator::ExportAsOptionAp214);

    stepCreator1->SetSettingsFile(strcat(UGII_BASE_DIR, "\\step214ug\\ugstep214.def"));

    stepCreator1->ExportSelectionBlock()->SetSelectionScope(NXOpen::ObjectSelector::ScopeSelectedObjects);

    stepCreator1->SetBsplineTol(0.0001);

    stepCreator1->SetInputFile(PARTNAME);

    int num_points = sample_num;
    std::vector<NXOpen::NXObject*> objects1(num_points);
    NXOpen::Features::PointSet* pointSet1(dynamic_cast<NXOpen::Features::PointSet*>(nXObject1));
    for (int i = 0; i < objects1.size(); ++i) {
        string point_name = "POINT " + std::to_string(i + 1);
        NXOpen::Point* point(dynamic_cast<NXOpen::Point*>(pointSet1->FindObject(point_name)));
        objects1[i] = point;
    }
    bool added1;
    added1 = stepCreator1->ExportSelectionBlock()->SelectionComp()->Add(objects1);

    stepCreator1->SetOutputFile(save_path.c_str());

    stepCreator1->SetFileSaveFlag(false);

    stepCreator1->SetLayerMask("1-256");

    stepCreator1->SetProcessHoldFlag(true);

    NXOpen::NXObject* nXObject2;
    nXObject2 = stepCreator1->Commit();

    stepCreator1->Destroy();
}

static void do_it(string section_path, string guide_path, string dir_path, string save_path, string feature_line_from_ug_0, string feature_line_from_ug_1, int sample_num, int degree)
{
    Session* theSession = Session::GetSession();
    string PARTNAME = std::to_string(rand() % 100000) + ".prt";
    doCreatePart(theSession, PARTNAME);
    Part* workPart(theSession->Parts()->Work());
    Part* displayPart(theSession->Parts()->Display());
    auto sections = getSectionLine(theSession, workPart, section_path);
    Point3d helpPointGuide, helpPointDir;
    auto guide = doCurveFit(workPart, displayPart, guide_path, helpPointGuide, degree);
    auto dir = doCurveFit(workPart, displayPart, dir_path, helpPointDir, degree);
    doStyledSweepwithDir(theSession, workPart, displayPart, sections, guide, helpPointGuide, dir, helpPointDir, save_path, PARTNAME);
    doSampleCurve(theSession, workPart, displayPart, guide, feature_line_from_ug_0, sample_num, PARTNAME);
    doSampleCurve(theSession, workPart, displayPart, dir, feature_line_from_ug_1, sample_num, PARTNAME);

}

int main(int argc, char** argv)
{
    string static_dir = argv[1];        // 第一个参数： static文件夹的路径，路径后面要带一个 /
    string recursion_dir = argv[2];     // 第二个参数： recursion文件夹的路径，路径后面要带一个 /
    int sample_num = 20000, degree = 5;
    if (4 >= argc) sample_num = atoi(argv[3]);
    if (5 == argc) degree = atoi(argv[4]);
    string section_path = static_dir + "mould_section.stp";  // 模具截面
    string guide_path = recursion_dir + "feature_line_for_ug_0.txt"; // 
    string dir_path = recursion_dir + "feature_line_for_ug_1.txt";
    string save_path = recursion_dir + "mould.stp";
    // 还需要输出 feature_line_for_ug 拟合之后的线再重采样一次的结果，输出到以下路径
    string feature_line_from_ug_0 = recursion_dir + "feature_line_from_ug_0.stp";
    string feature_line_from_ug_1 = recursion_dir + "feature_line_from_ug_1.stp";

    cout << argc << endl;
    cout << section_path << endl;
    cout << guide_path << endl;
    cout << dir_path << endl;
    cout << save_path << endl;

    UGII_BASE_DIR = getenv("UGII_BASE_DIR");

    int cnt = 0;
    while ((-1 == (_access(feature_line_from_ug_0.c_str(), 0)) || -1 == (_access(feature_line_from_ug_1.c_str(), 0))) && cnt < 3) {
        try {
            cout << "begin!" << endl;
            do_it(section_path, guide_path, dir_path, save_path, feature_line_from_ug_0, feature_line_from_ug_1, sample_num, degree);
            cout << "end!" << endl;
        }
        catch(exception& e){
            cnt++;
        }
    }
    if (cnt >= 3) cout << "errors' happened in data files.\n";
    return 0;
}



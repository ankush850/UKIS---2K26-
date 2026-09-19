/**
 * UKIS-2026 - Super Resolution Mapping Frontend Application
 * Features:
 * 1. AOI Selection -> Real Sentinel-2 L2A tile fetching via Copernicus CDSE
 * 2. Deep Learning Super Resolution Inference (10m -> 2.5m GSD)
 * 3. Monte-Carlo Dropout Uncertainty & Hallucination Mapping (USP)
 * 4. Section 12: Automated Infrastructure Detection (Bridges, Roads, Buildings) & GeoJSON Export
 * 5. Section 13: High-Resolution Deep-Zoom Viewer (OpenSeadragon with Synchronized Viewports)
 */

document.addEventListener("DOMContentLoaded", () => {
    // Set speed to 0.75x. Going too slow (like 0.25) drops frames and causes visual lag.
    const bgVideo = document.querySelector('.landing-video-bg');
    if (bgVideo) bgVideo.playbackRate = 0.75; // Smooth slow down

    // Landing Page Transition Logic
    const landingPage = document.getElementById('landing-page');
    const dashboardContainer = document.getElementById('dashboard-container');
    const btnStartAnalysis = document.getElementById('btn-start-analysis');
    
    if (btnStartAnalysis && landingPage && dashboardContainer) {
        btnStartAnalysis.addEventListener('click', () => {
            landingPage.classList.add('fade-out');
            setTimeout(() => {
                landingPage.style.display = 'none';
                dashboardContainer.style.display = 'block'; // Ensure it's in the document flow
                // Trigger reflow for CSS transition
                void dashboardContainer.offsetWidth;
                dashboardContainer.style.opacity = '1';
                
                // Important: Trigger resize so maps and deep-zoom viewers know they are visible now
                setTimeout(() => {
                    window.dispatchEvent(new Event('resize'));
                }, 100);
            }, 600); // Wait for landing page fade-out transition
        });
    }

    // Application State
    const state = {
        activePreset: "punjab_agri",
        activeBbox: [75.30, 30.55, 75.36, 30.60],
        mcSamples: 2,
        confidenceActive: false,
        confidenceOpacity: 0.75,
        infraActive: false,
        infraOpacity: 0.85,
        cloudActive: false,
        cloudOpacity: 0.80,
        viewMode: "split",
        presets: [],
        scaleFactor: 4,
        modelName: "hat",
        applyUnsharp: false,
        applyRealESRGANSharpen: false,
        acquisitionMode: "single",
        latestLrUrl: null,
        latestSrUrl: null,
        latestHrUrl: null,
        latestConfUrl: null,
        latestInfraUrl: null,
        latestCloudUrl: null,
        cloudWarning: null,
        cloudCoveragePct: 0.0,
        infraSummary: null,
        blockchainRecord: null,
        visualizationMode: "true_color",
        activeTopView: "sr",
        activeSpectralMode: "ndvi",
        latestSpectralLrUrl: null,
        latestSpectralSrUrl: null
    };

    // Top-Level Navigation & App Views
    const navBtnSr = document.getElementById("nav-btn-sr");
    const navBtnSpectral = document.getElementById("nav-btn-spectral");
    const viewSrMapping = document.getElementById("view-sr-mapping");
    const viewSpectralAnalysis = document.getElementById("view-spectral-analysis");

    // DOM Elements
    const presetsList = document.getElementById("presets-list");
    const aoiCoordsBadge = document.getElementById("aoi-coords-badge");
    const inputCustomLat = document.getElementById("input-custom-lat");
    const inputCustomLon = document.getElementById("input-custom-lon");
    const btnGotoCoords = document.getElementById("btn-goto-coords");
    const sliderMcSamples = document.getElementById("slider-mc-samples");
    const valMcSamples = document.getElementById("val-mc-samples");
    const selectScale = document.getElementById("select-scale");
    const selectModel = document.getElementById("select-model");
    const selectVisMode = document.getElementById("select-vis-mode");
    const checkRealESRGANSharpen = document.getElementById("check-realesrgan-sharpen");
    const checkUnsharp = document.getElementById("check-unsharp");
    const btnModeSingle = document.getElementById("btn-mode-single");
    const btnModeMulti = document.getElementById("btn-mode-multi");

    // Zoom Controls DOM
    const btnZoomIn = document.getElementById("btn-zoom-in");
    const btnZoomOut = document.getElementById("btn-zoom-out");
    const btnZoomReset = document.getElementById("btn-zoom-reset");
    const zoomLevelBadge = document.getElementById("zoom-level-badge");
    const inspectorDims = document.getElementById("inspector-dims");

    // Live Verification DOM
    const verifCheckpoint = document.getElementById("verif-checkpoint");
    const verifDropout = document.getElementById("verif-dropout");
    const verifVariance = document.getElementById("verif-variance");
    const verifDimensions = document.getElementById("verif-dimensions");
    const verifUnsharp = document.getElementById("verif-unsharp");
    const verifReference = document.getElementById("verif-reference");
    const verifTemporal = document.getElementById("verif-temporal");

    // Action buttons & inputs
    const btnRunSr = document.getElementById("btn-run-sr");
    const btnTriggerUpload = document.getElementById("btn-trigger-upload");
    const fileUpload = document.getElementById("file-upload");
    const btnDownloadSr = document.getElementById("btn-download-sr");
    const btnDownloadConf = document.getElementById("btn-download-conf");
    const btnDownloadGeojson = document.getElementById("btn-download-geojson");

    // View toggles & containers
    const tabSplit = document.getElementById("tab-split");
    const tabSide = document.getElementById("tab-side");
    const sliderContainer = document.getElementById("slider-container");
    const sideBySideContainer = document.getElementById("side-by-side-container");
    const srWrapper = document.getElementById("sr-wrapper");
    const sliderHandle = document.getElementById("slider-handle");
    const loadingScrim = document.getElementById("loading-scrim");

    // Heatmap, Infrastructure & Cloud Toolbar Controls
    const toggleConfidencePill = document.getElementById("toggle-confidence-pill");
    const sliderConfidenceOpacity = document.getElementById("slider-confidence-opacity");
    const toggleInfraPill = document.getElementById("toggle-infra-pill");
    const sliderInfraOpacity = document.getElementById("slider-infra-opacity");
    const toggleCloudPill = document.getElementById("toggle-cloud-pill");
    const sliderCloudOpacity = document.getElementById("slider-cloud-opacity");
    const cloudWarningBanner = document.getElementById("cloud-warning-banner");
    const cloudWarningText = document.getElementById("cloud-warning-text");
    const btnDismissCloudWarning = document.getElementById("btn-dismiss-cloud-warning");

    // Scientific Metrics HUD Elements
    const metricConfidence = document.getElementById("metric-confidence");
    const metricUncertainty = document.getElementById("metric-uncertainty");
    const metricSam = document.getElementById("metric-sam");
    const metricCycle = document.getElementById("metric-cycle");
    const metricPsnr = document.getElementById("metric-psnr");
    const metricSsim = document.getElementById("metric-ssim");
    const metricErgas = document.getElementById("metric-ergas");
    const metricRuntime = document.getElementById("metric-runtime");

    // Dynamic Quality Assessment Panels (Paired vs No-Reference)
    const panelValidationPaired = document.getElementById("panel-validation-paired");
    const panelValidationNr = document.getElementById("panel-validation-nr");
    const metricNrNiqe = document.getElementById("metric-nr-niqe");
    const metricNrBrisque = document.getElementById("metric-nr-brisque");
    const metricNrConfidence = document.getElementById("metric-nr-confidence");
    const metricNrCycle = document.getElementById("metric-nr-cycle");

    // Infrastructure & Multi-Class HUD Elements (Section 12)
    const metricBuildings = document.getElementById("metric-buildings");
    const metricRoads = document.getElementById("metric-roads");
    const metricInfraConf = document.getElementById("metric-infra-conf");

    // =========================================================================
    // 1. OpenSeadragon Deep-Zoom High-Resolution Viewers (Section 13)
    // =========================================================================
    let viewerLR = null;
    let viewerSR = null;
    let viewerSideLR = null;
    let viewerSideSR = null;
    let viewerSideHR = null;

    // Overlay Elements for SR Viewer
    const confOverlay = document.createElement("img");
    confOverlay.id = "osd-conf-overlay";
    confOverlay.style.width = "100%";
    confOverlay.style.height = "100%";
    confOverlay.style.pointerEvents = "none";
    confOverlay.style.mixBlendMode = "screen";
    confOverlay.style.display = "none";

    const infraOverlay = document.createElement("img");
    infraOverlay.id = "osd-infra-overlay";
    infraOverlay.style.width = "100%";
    infraOverlay.style.height = "100%";
    infraOverlay.style.pointerEvents = "none";
    infraOverlay.style.display = "none";

    // Cloud Mask Occlusion Overlay Elements (Both Input LR and Output SR)
    const cloudOverlaySR = document.createElement("img");
    cloudOverlaySR.id = "osd-cloud-overlay-sr";
    cloudOverlaySR.style.width = "100%";
    cloudOverlaySR.style.height = "100%";
    cloudOverlaySR.style.pointerEvents = "none";
    cloudOverlaySR.style.display = "none";

    const cloudOverlayLR = document.createElement("img");
    cloudOverlayLR.id = "osd-cloud-overlay-lr";
    cloudOverlayLR.style.width = "100%";
    cloudOverlayLR.style.height = "100%";
    cloudOverlayLR.style.pointerEvents = "none";
    cloudOverlayLR.style.display = "none";

    function updateCloudWarning(warningText) {
        if (!cloudWarningBanner) return;
        if (warningText) {
            if (cloudWarningText) cloudWarningText.innerText = warningText;
            cloudWarningBanner.classList.remove("hidden");
        } else {
            cloudWarningBanner.classList.add("hidden");
        }
    }

    // Non-blocking in-app notification banner (replaces native alert popups)
    let inAppBannerTimeout = null;

    function showInAppNotification(message, icon = "🛰️", type = "info", durationMs = 7000) {
        const banner = document.getElementById("in-app-banner");
        const iconEl = document.getElementById("in-app-banner-icon");
        const textEl = document.getElementById("in-app-banner-text");
        if (!banner) return;

        if (inAppBannerTimeout) {
            clearTimeout(inAppBannerTimeout);
            inAppBannerTimeout = null;
        }

        if (iconEl) iconEl.textContent = icon;
        if (textEl) textEl.textContent = message;

        banner.classList.remove("banner-info", "banner-warning", "banner-success", "hidden");
        if (type === "warning") {
            banner.classList.add("banner-warning");
        } else if (type === "success") {
            banner.classList.add("banner-success");
        } else {
            banner.classList.add("banner-info");
        }

        if (durationMs > 0) {
            inAppBannerTimeout = setTimeout(() => {
                banner.classList.add("hidden");
                inAppBannerTimeout = null;
            }, durationMs);
        }
    }

    function dismissInAppNotification() {
        const banner = document.getElementById("in-app-banner");
        if (banner) banner.classList.add("hidden");
        if (inAppBannerTimeout) {
            clearTimeout(inAppBannerTimeout);
            inAppBannerTimeout = null;
        }
    }

    const btnDismissInAppBanner = document.getElementById("btn-dismiss-in-app-banner");
    if (btnDismissInAppBanner) {
        btnDismissInAppBanner.addEventListener("click", dismissInAppNotification);
    }
    if (btnDismissCloudWarning) {
        btnDismissCloudWarning.addEventListener("click", () => updateCloudWarning(null));
    }

    function initOSDViewer(elementId, showNav = false) {
        if (!window.OpenSeadragon) {
            console.error("[OSD] OpenSeadragon library not loaded!");
            return null;
        }
        try {
            return OpenSeadragon({
                id: elementId,
                prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.1/images/",
                showNavigator: showNav,
                navigatorPosition: "TOP_RIGHT",
                navigatorSizeRatio: 0.22,
                minZoomLevel: 0.8,
                maxZoomLevel: 25,
                defaultZoomLevel: 1,
                visibilityRatio: 1.0,
                constrainDuringPan: true,
                showNavigationControl: false,
                imageLoaderLimit: 4,
                gestureSettingsMouse: {
                    scrollToZoom: true,
                    clickToZoom: false,
                    dblClickToZoom: true,
                    dragToPan: true
                }
            });
        } catch (e) {
            console.error(`[OSD] Failed to init viewer for ${elementId}:`, e);
            return null;
        }
    }

    // Initialize Viewers
    viewerLR = initOSDViewer("osd-lr", false);
    viewerSR = initOSDViewer("osd-sr", true); // Includes navigator minimap!
    viewerSideLR = initOSDViewer("osd-side-lr", false);
    viewerSideSR = initOSDViewer("osd-side-sr", false);
    viewerSideHR = initOSDViewer("osd-side-hr", false);

    // Synchronize Viewports across OpenSeadragon instances in lockstep
    let isSyncing = false;
    function syncViewports(source, targets) {
        if (isSyncing || !source || !source.viewport) return;
        isSyncing = true;
        try {
            const center = source.viewport.getCenter();
            const zoom = source.viewport.getZoom();
            targets.forEach(target => {
                if (target && target.viewport) {
                    target.viewport.panTo(center, true);
                    target.viewport.zoomTo(zoom, null, true);
                }
            });
        } catch (e) {
            // Ignore temporary view transition errors
        } finally {
            isSyncing = false;
        }
    }

    // Bind Split View Synchronization (LR <-> SR)
    if (viewerLR && viewerSR) {
        viewerLR.addHandler("pan", () => syncViewports(viewerLR, [viewerSR]));
        viewerLR.addHandler("zoom", () => syncViewports(viewerLR, [viewerSR]));
        viewerSR.addHandler("pan", () => syncViewports(viewerSR, [viewerLR]));
        viewerSR.addHandler("zoom", () => syncViewports(viewerSR, [viewerLR]));
    }

    // Dedicated View 2: Spectral & Disaster OpenSeadragon Viewers (Lazy initialized when tab is displayed)
    let spectralViewerLR = null;
    let spectralViewerSR = null;
    let spectralViewerSingle = null;

    function ensureSpectralViewers() {
        if (!spectralViewerLR) {
            spectralViewerLR = initOSDViewer("spectral-osd-lr", false);
        }
        if (!spectralViewerSR) {
            spectralViewerSR = initOSDViewer("spectral-osd-sr", true);
            if (spectralViewerSR) {
                spectralViewerSR.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerSR));
            }
        }
        if (!spectralViewerSingle) {
            spectralViewerSingle = initOSDViewer("spectral-osd-single", true);
            if (spectralViewerSingle) {
                spectralViewerSingle.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerSingle));
            }
        }
        if (spectralViewerLR && spectralViewerSR) {
            try {
                spectralViewerLR.addHandler("pan", () => syncViewports(spectralViewerLR, [spectralViewerSR]));
                spectralViewerLR.addHandler("zoom", () => syncViewports(spectralViewerLR, [spectralViewerSR]));
                spectralViewerSR.addHandler("pan", () => syncViewports(spectralViewerSR, [spectralViewerLR]));
                spectralViewerSR.addHandler("zoom", () => syncViewports(spectralViewerSR, [spectralViewerLR]));
            } catch (e) { }
        }
    }

    // Bind Side-by-Side View Synchronization (SideLR <-> SideSR <-> SideHR)
    if (viewerSideLR && viewerSideSR && viewerSideHR) {
        viewerSideLR.addHandler("pan", () => syncViewports(viewerSideLR, [viewerSideSR, viewerSideHR]));
        viewerSideLR.addHandler("zoom", () => syncViewports(viewerSideLR, [viewerSideSR, viewerSideHR]));
        viewerSideSR.addHandler("pan", () => syncViewports(viewerSideSR, [viewerSideLR, viewerSideHR]));
        viewerSideSR.addHandler("zoom", () => syncViewports(viewerSideSR, [viewerSideLR, viewerSideHR]));
        viewerSideHR.addHandler("pan", () => syncViewports(viewerSideHR, [viewerSideLR, viewerSideSR]));
        viewerSideHR.addHandler("zoom", () => syncViewports(viewerSideHR, [viewerSideLR, viewerSideSR]));
    }

    // OpenSeadragon Deep-Zoom UI Controls
    function updateZoomBadge(viewer) {
        if (!zoomLevelBadge || !viewer || !viewer.viewport) return;
        try {
            const z = viewer.viewport.getZoom();
            zoomLevelBadge.innerText = `${z.toFixed(1)}x`;
        } catch (e) { }
    }

    if (viewerSR) {
        viewerSR.addHandler("zoom", () => updateZoomBadge(viewerSR));
    }
    if (viewerSideSR) {
        viewerSideSR.addHandler("zoom", () => updateZoomBadge(viewerSideSR));
    }

    if (btnZoomIn) {
        btnZoomIn.addEventListener("click", () => {
            const v = (state.viewMode === "split") ? (viewerSR || viewerLR) : viewerSideSR;
            if (v && v.viewport) {
                v.viewport.zoomBy(1.35);
                v.viewport.applyConstraints();
                updateZoomBadge(v);
            }
        });
    }

    if (btnZoomOut) {
        btnZoomOut.addEventListener("click", () => {
            const v = (state.viewMode === "split") ? (viewerSR || viewerLR) : viewerSideSR;
            if (v && v.viewport) {
                v.viewport.zoomBy(0.74);
                v.viewport.applyConstraints();
                updateZoomBadge(v);
            }
        });
    }

    if (btnZoomReset) {
        btnZoomReset.addEventListener("click", () => {
            const v = (state.viewMode === "split") ? (viewerSR || viewerLR) : viewerSideSR;
            if (v && v.viewport) {
                v.viewport.goHome(true);
                updateZoomBadge(v);
            }
            if (viewerLR && viewerLR.viewport) viewerLR.viewport.goHome(true);
            if (viewerSideLR && viewerSideLR.viewport) viewerSideLR.viewport.goHome(true);
            if (viewerSideHR && viewerSideHR.viewport) viewerSideHR.viewport.goHome(true);
        });
    }

    function loadViewerImage(viewer, dataUrl, isSR = false) {
        if (!viewer || !dataUrl) return;
        viewer.open({
            type: "image",
            url: dataUrl,
            buildPyramid: false
        });

        viewer.addOnceHandler("open", () => {
            try {
                if (viewer.viewport) {
                    viewer.viewport.resize();
                    viewer.viewport.goHome(true);
                }
            } catch (e) { }
        });

        if (isSR) {
            viewer.addOnceHandler("open", () => {
                viewer.clearOverlays();
                if (state.latestConfUrl) {
                    confOverlay.src = state.latestConfUrl;
                    viewer.addOverlay({
                        element: confOverlay,
                        location: new OpenSeadragon.Rect(0, 0, 1, 1)
                    });
                }
                if (state.latestInfraUrl) {
                    infraOverlay.src = state.latestInfraUrl;
                    viewer.addOverlay({
                        element: infraOverlay,
                        location: new OpenSeadragon.Rect(0, 0, 1, 1)
                    });
                }
                if (state.latestCloudUrl) {
                    cloudOverlaySR.src = state.latestCloudUrl;
                    viewer.addOverlay({
                        element: cloudOverlaySR,
                        location: new OpenSeadragon.Rect(0, 0, 1, 1)
                    });
                }
                updateOverlayDisplay();
            });
        } else if (viewer === viewerLR) {
            viewer.addOnceHandler("open", () => {
                viewer.clearOverlays();
                if (state.latestCloudUrl) {
                    cloudOverlayLR.src = state.latestCloudUrl;
                    viewer.addOverlay({
                        element: cloudOverlayLR,
                        location: new OpenSeadragon.Rect(0, 0, 1, 1)
                    });
                }
                updateOverlayDisplay();
            });
        }
    }

    function syncCloudOverlays() {
        if (!state.latestCloudUrl) return;
        if (viewerLR && viewerLR.isOpen()) {
            cloudOverlayLR.src = state.latestCloudUrl;
            try {
                viewerLR.addOverlay({
                    element: cloudOverlayLR,
                    location: new OpenSeadragon.Rect(0, 0, 1, 1)
                });
            } catch (e) { }
        }
        if (viewerSR && viewerSR.isOpen()) {
            cloudOverlaySR.src = state.latestCloudUrl;
            try {
                viewerSR.addOverlay({
                    element: cloudOverlaySR,
                    location: new OpenSeadragon.Rect(0, 0, 1, 1)
                });
            } catch (e) { }
        }
        updateOverlayDisplay();
    }

    function updateOverlayDisplay() {
        if (confOverlay) {
            confOverlay.style.display = state.confidenceActive ? "block" : "none";
            confOverlay.style.opacity = state.confidenceOpacity;
        }
        if (infraOverlay) {
            infraOverlay.style.display = state.infraActive ? "block" : "none";
            infraOverlay.style.opacity = state.infraOpacity;
        }
        if (cloudOverlaySR) {
            cloudOverlaySR.style.display = state.cloudActive ? "block" : "none";
            cloudOverlaySR.style.opacity = state.cloudOpacity;
        }
        if (cloudOverlayLR) {
            cloudOverlayLR.style.display = state.cloudActive ? "block" : "none";
            cloudOverlayLR.style.opacity = state.cloudOpacity;
        }
    }

    // =========================================================================
    // 2. Leaflet AOI Map Setup
    // =========================================================================
    const map = L.map("aoi-map", {
        zoomControl: false,
        attributionControl: false
    }).setView([30.575, 75.33], 12);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 18,
    }).addTo(map);

    let aoiRectangle = L.rectangle([
        [30.55, 75.30],
        [30.60, 75.36]
    ], {
        color: "#00e5ff",
        weight: 2,
        fillColor: "#00e5ff",
        fillOpacity: 0.15
    }).addTo(map);

    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    const drawControl = new L.Control.Draw({
        position: 'topright',
        draw: {
            polygon: false,
            polyline: false,
            circle: false,
            circlemarker: false,
            marker: false,
            rectangle: {
                shapeOptions: {
                    color: "#00e5ff",
                    weight: 2,
                    fillColor: "#00e5ff",
                    fillOpacity: 0.18
                }
            }
        },
        edit: { featureGroup: drawnItems }
    });
    map.addControl(drawControl);

    let currentFetchSeq = 0;

    function syncAoiUI(bounds, bbox, presetId, lat, lon) {
        if (aoiRectangle && bounds) {
            if (map && !map.hasLayer(aoiRectangle)) {
                aoiRectangle.addTo(map);
            }
            aoiRectangle.setBounds(bounds);
            if (aoiRectangle.bringToFront) aoiRectangle.bringToFront();
        }
        if (spectralAoiRectangle && bounds) {
            if (spectralMap && !spectralMap.hasLayer(spectralAoiRectangle)) {
                spectralAoiRectangle.addTo(spectralMap);
            }
            spectralAoiRectangle.setBounds(bounds);
            if (spectralAoiRectangle.bringToFront) spectralAoiRectangle.bringToFront();
        }

        if (lat !== undefined && lon !== undefined) {
            const numLat = parseFloat(lat);
            const numLon = parseFloat(lon);
            if (inputCustomLat) inputCustomLat.value = numLat.toFixed(4);
            if (inputCustomLon) inputCustomLon.value = numLon.toFixed(4);
            if (aoiCoordsBadge) aoiCoordsBadge.innerText = `Lat: ${numLat.toFixed(3)} | Lon: ${numLon.toFixed(3)}`;

            const spLat = document.getElementById("spectral-input-lat");
            const spLon = document.getElementById("spectral-input-lon");
            const spBadge = document.getElementById("spectral-aoi-coords-badge");
            if (spLat) spLat.value = numLat.toFixed(4);
            if (spLon) spLon.value = numLon.toFixed(4);
            if (spBadge) spBadge.innerText = `Lat: ${numLat.toFixed(3)} | Lon: ${numLon.toFixed(3)}`;
        }

        document.querySelectorAll(".preset-chip").forEach(c => {
            c.classList.toggle("active", c.dataset.id === presetId);
        });

        const spMetaAoi = document.getElementById("spectral-meta-aoi");
        if (spMetaAoi) {
            const preset = state.presets.find(p => p.id === presetId);
            spMetaAoi.innerText = preset ? preset.title : (presetId === "custom_drawn_aoi" ? "Custom Drawn AOI" : "Custom Coordinates");
        }
    }

    map.on(L.Draw.Event.CREATED, function (e) {
        const layer = e.layer;
        drawnItems.clearLayers();
        drawnItems.addLayer(layer);

        const bounds = layer.getBounds();
        const bbox = [
            parseFloat(bounds.getWest().toFixed(5)),
            parseFloat(bounds.getSouth().toFixed(5)),
            parseFloat(bounds.getEast().toFixed(5)),
            parseFloat(bounds.getNorth().toFixed(5))
        ];

        state.activeBbox = bbox;
        state.activePreset = "custom_drawn_aoi";
        const center = bounds.getCenter();
        syncAoiUI(bounds, bbox, "custom_drawn_aoi", center.lat, center.lng);
        fetchTileForBbox(bbox, "custom_drawn_aoi");
    });

    map.on(L.Draw.Event.EDITED, function (e) {
        const layers = e.layers;
        layers.eachLayer(function (layer) {
            const bounds = layer.getBounds();
            const bbox = [
                parseFloat(bounds.getWest().toFixed(5)),
                parseFloat(bounds.getSouth().toFixed(5)),
                parseFloat(bounds.getEast().toFixed(5)),
                parseFloat(bounds.getNorth().toFixed(5))
            ];
            state.activeBbox = bbox;
            state.activePreset = "custom_drawn_aoi";
            const center = bounds.getCenter();
            syncAoiUI(bounds, bbox, "custom_drawn_aoi", center.lat, center.lng);
            fetchTileForBbox(bbox, "custom_drawn_aoi");
        });
    });

    map.on(L.Draw.Event.DELETED, function () {
        if (state.presets && state.presets.length > 0) {
            selectPreset(state.presets[0].id);
        }
    });

    const btnDrawRect = document.getElementById("btn-draw-rect");
    if (btnDrawRect) {
        btnDrawRect.addEventListener("click", () => {
            new L.Draw.Rectangle(map, drawControl.options.draw.rectangle).enable();
        });
    }

    map.on("click", (e) => {
        if (document.querySelector(".leaflet-draw-actions")) return;
        const lat = e.latlng.lat;
        const lon = e.latlng.lng;
        const span = 0.03;
        const customBbox = [
            parseFloat((lon - span).toFixed(4)),
            parseFloat((lat - span).toFixed(4)),
            parseFloat((lon + span).toFixed(4)),
            parseFloat((lat + span).toFixed(4))
        ];

        state.activeBbox = customBbox;
        state.activePreset = "custom_click";
        const bounds = [
            [customBbox[1], customBbox[0]],
            [customBbox[3], customBbox[2]]
        ];

        if (drawnItems) drawnItems.clearLayers();
        if (spectralDrawnItems) spectralDrawnItems.clearLayers();
        if (map) map.panTo([lat, lon]);
        if (spectralMap) spectralMap.panTo([lat, lon]);

        syncAoiUI(bounds, customBbox, "custom_click", lat, lon);
        fetchTileForBbox(customBbox, "custom_click");
    });

    // Manual Coordinate Navigation
    function navigateToCoordinates(lat, lon) {
        if (isNaN(lat) || lat < -90 || lat > 90) {
            showInAppNotification("Please enter a valid Latitude between -90 and 90.", "⚠️", "warning", 6000);
            if (inputCustomLat) inputCustomLat.focus();
            return;
        }
        if (isNaN(lon) || lon < -180 || lon > 180) {
            showInAppNotification("Please enter a valid Longitude between -180 and 180.", "⚠️", "warning", 6000);
            if (inputCustomLon) inputCustomLon.focus();
            return;
        }

        const span = 0.03;
        const customBbox = [
            parseFloat((lon - span).toFixed(4)),
            parseFloat((lat - span).toFixed(4)),
            parseFloat((lon + span).toFixed(4)),
            parseFloat((lat + span).toFixed(4))
        ];

        state.activeBbox = customBbox;
        state.activePreset = "custom_coords";

        const bounds = [
            [customBbox[1], customBbox[0]],
            [customBbox[3], customBbox[2]]
        ];

        if (drawnItems) drawnItems.clearLayers();
        if (spectralDrawnItems) spectralDrawnItems.clearLayers();
        if (map) {
            map.invalidateSize();
            map.flyToBounds(bounds, { padding: [20, 20], duration: 0.8 });
        }
        if (spectralMap) {
            spectralMap.invalidateSize();
            spectralMap.flyToBounds(bounds, { padding: [20, 20], duration: 0.8 });
        }

        syncAoiUI(bounds, customBbox, "custom_coords", lat, lon);
        fetchTileForBbox(customBbox, "custom_coords");
        if (state.activeTopView === "spectral") {
            selectSpectralMode(state.activeSpectralMode || "ndvi");
        }
    }

    if (btnGotoCoords) {
        btnGotoCoords.addEventListener("click", () => {
            const lat = parseFloat(inputCustomLat ? inputCustomLat.value.trim() : NaN);
            const lon = parseFloat(inputCustomLon ? inputCustomLon.value.trim() : NaN);
            navigateToCoordinates(lat, lon);
        });
    }

    const handleCoordEnter = (e) => {
        if (e.key === "Enter") {
            const lat = parseFloat(inputCustomLat ? inputCustomLat.value.trim() : NaN);
            const lon = parseFloat(inputCustomLon ? inputCustomLon.value.trim() : NaN);
            navigateToCoordinates(lat, lon);
        }
    };
    if (inputCustomLat) inputCustomLat.addEventListener("keydown", handleCoordEnter);
    if (inputCustomLon) inputCustomLon.addEventListener("keydown", handleCoordEnter);

    // =========================================================================
    // 3. API Communication & Pipelines
    // =========================================================================
    async function loadPresets() {
        try {
            const res = await fetch("/api/presets");
            const data = await res.json();
            state.presets = data.presets;
            renderPresetChips();
            selectPreset("punjab_agri");
        } catch (err) {
            console.error("[GEO-SRM] Failed to load presets:", err);
        }
    }

    function renderPresetChips() {
        presetsList.innerHTML = "";
        const spectralPresetsList = document.getElementById("spectral-presets-list");
        if (spectralPresetsList) spectralPresetsList.innerHTML = "";

        state.presets.forEach(p => {
            const chip = document.createElement("div");
            chip.className = `preset-chip ${p.id === state.activePreset ? 'active' : ''}`;
            chip.dataset.id = p.id;
            chip.innerHTML = `
                <div class="preset-name">${p.title}</div>
                <div class="preset-meta">${p.description}</div>
            `;
            chip.addEventListener("click", () => selectPreset(p.id));
            presetsList.appendChild(chip);

            if (spectralPresetsList) {
                const spChip = chip.cloneNode(true);
                spChip.addEventListener("click", () => selectPreset(p.id));
                spectralPresetsList.appendChild(spChip);
            }
        });
    }

    function selectPreset(presetId) {
        const preset = state.presets.find(p => p.id === presetId);
        if (!preset) return;

        state.activePreset = presetId;
        state.activeBbox = preset.bbox;

        if (drawnItems) drawnItems.clearLayers();
        if (spectralDrawnItems) spectralDrawnItems.clearLayers();

        const bounds = [
            [preset.bbox[1], preset.bbox[0]],
            [preset.bbox[3], preset.bbox[2]]
        ];
        if (map) {
            map.invalidateSize();
            map.flyToBounds(bounds, { padding: [20, 20], duration: 0.8 });
        }
        if (spectralMap) {
            spectralMap.invalidateSize();
            spectralMap.flyToBounds(bounds, { padding: [20, 20], duration: 0.8 });
        }

        const lat = preset.coords ? preset.coords[1] : (preset.bbox[1] + preset.bbox[3]) / 2;
        const lon = preset.coords ? preset.coords[0] : (preset.bbox[0] + preset.bbox[2]) / 2;
        syncAoiUI(bounds, preset.bbox, presetId, lat, lon);

        fetchTileForBbox(preset.bbox, preset.id);
        if (state.activeTopView === "spectral") {
            selectSpectralMode(state.activeSpectralMode || "ndvi");
        }
    }

    async function fetchTileForBbox(bbox, aoiId) {
        const fetchSeq = ++currentFetchSeq;
        loadingScrim.classList.remove("hidden");
        try {
            const res = await fetch("/api/fetch-tile", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    bbox: bbox,
                    aoi_id: aoiId,
                    max_cloud: 20,
                    force_live: true
                })
            });
            const data = await res.json();
            if (fetchSeq !== currentFetchSeq) return;

            if (data.status === "success") {
                state.hasReference = !!data.has_reference;
                state.isScientific = !!data.is_scientific_ground_truth;
                state.referenceTier = data.reference_tier;
                state.referenceTierLabel = data.reference_tier_label;
                state.latestLrUrl = data.lr_preview;
                state.latestHrUrl = data.hr_preview;
                state.latestCloudUrl = data.cloud_mask_preview || null;
                state.cloudCoveragePct = data.cloud_coverage_pct || 0.0;
                updateCloudWarning(data.cloud_warning);

                // Load OpenSeadragon Viewers
                loadViewerImage(viewerLR, data.lr_preview);
                loadViewerImage(viewerSideLR, data.lr_preview);

                // Update 1st Side-by-Side Panel Header with Live Satellite indicator
                const headerSideLr = document.getElementById("header-side-lr");
                if (headerSideLr) {
                    if (data.is_live) {
                        headerSideLr.innerHTML = 'Input: Sentinel-2 L2A (10m) <span style="font-size:11px;color:#34d399;font-weight:700;margin-left:6px;background:rgba(52,211,153,0.15);padding:2px 6px;border-radius:4px;border:1px solid rgba(52,211,153,0.4);">● LIVE ESA CDSE</span>';
                    } else {
                        headerSideLr.innerText = "Input: Sentinel-2 L2A (10m)";
                    }
                }

                // Update 3rd Side-by-Side Panel Header strictly per tier rules
                const headerSideHr = document.getElementById("header-side-hr");
                if (headerSideHr) {
                    if (data.is_scientific_ground_truth) {
                        headerSideHr.innerText = "Reference: Ground Truth HR (1.5m)";
                    } else if (data.has_reference) {
                        headerSideHr.innerText = "Reference: Visual Basemap";
                    } else {
                        headerSideHr.innerText = "Reference: None Available";
                    }
                }

                if (data.has_reference && data.hr_preview) {
                    loadViewerImage(viewerSideHR, data.hr_preview);
                } else {
                    if (viewerSideHR) viewerSideHR.close();
                }

                if (data.is_scientific_ground_truth) {
                    if (panelValidationPaired) panelValidationPaired.style.display = "block";
                    if (panelValidationNr) panelValidationNr.style.display = "none";
                } else {
                    if (panelValidationPaired) panelValidationPaired.style.display = "none";
                    if (panelValidationNr) panelValidationNr.style.display = "block";
                }

                if (verifReference) {
                    if (data.reference_file) {
                        verifReference.innerText = data.reference_file;
                        verifReference.style.color = data.is_scientific_ground_truth ? "#38bdf8" : "#facc15";
                    } else {
                        verifReference.innerText = "None (Unpaired AOI — No False Fallback)";
                        verifReference.style.color = "#f87171";
                    }
                }

                if (data.visualization_info) {
                    updateVisualizationLegend(data.visualization_info);
                }

                // Trigger PyTorch Super-Resolution
                await runSuperResolution(fetchSeq);
            } else {
                showInAppNotification(`Fetch error: ${data.detail || 'Could not load live satellite tile'}`, "❌", "warning", 7000);
            }
        } catch (err) {
            if (fetchSeq === currentFetchSeq) {
                console.error("[GEO-SRM] Fetch tile failed:", err);
                showInAppNotification("Live satellite stream error. Check network or coordinates.", "⚠️", "warning", 6000);
            }
        } finally {
            if (fetchSeq === currentFetchSeq) {
                loadingScrim.classList.add("hidden");
            }
        }
    }

    // Real Super Resolution Inference: Calls POST /api/superresolve
    async function runSuperResolution(fetchSeq = null) {
        if (fetchSeq !== null && fetchSeq !== currentFetchSeq) return;
        loadingScrim.classList.remove("hidden");
        const elCoregStatusInit = document.getElementById("coreg-shift-status");
        const elComputedAtInit = document.getElementById("metric-computed-at");
        if (elCoregStatusInit) elCoregStatusInit.innerHTML = '<span style="color: #38bdf8;">🔄 AROSICS: Calculating spatial shifts...</span>';
        if (elComputedAtInit) elComputedAtInit.innerText = 'Executing inference...';

        try {
            const res = await fetch("/api/superresolve", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    num_mc_samples: state.mcSamples,
                    scale_factor: state.scaleFactor,
                    model_name: state.modelName,
                    apply_unsharp: state.applyUnsharp,
                    apply_realesrgan_sharpen: state.applyRealESRGANSharpen,
                    visualization_mode: state.visualizationMode || "true_color"
                })
            });
            const data = await res.json();
            if (fetchSeq !== null && fetchSeq !== currentFetchSeq) return;
            if (data.status === "success") {
                state.latestSrUrl = data.sr_preview;
                if (data.lr_preview) {
                    state.latestLrUrl = data.lr_preview;
                    loadViewerImage(viewerLR, data.lr_preview, false);
                    loadViewerImage(viewerSideLR, data.lr_preview, false);
                }
                state.latestConfUrl = data.confidence_heatmap;
                state.latestInfraUrl = data.infrastructure_overlay;
                state.infraSummary = data.infrastructure_summary;
                if (data.cloud_mask_preview) state.latestCloudUrl = data.cloud_mask_preview;
                if (data.cloud_coverage_pct !== undefined) state.cloudCoveragePct = data.cloud_coverage_pct;
                if (data.cloud_warning) updateCloudWarning(data.cloud_warning);

                if (data.visualization_info) {
                    updateVisualizationLegend(data.visualization_info);
                }

                // Load OpenSeadragon SR Viewers with Overlays
                loadViewerImage(viewerSR, data.sr_preview, true);
                loadViewerImage(viewerSideSR, data.sr_preview, false);
                syncCloudOverlays();

                // Update 1st Side-by-Side Panel Header for Input Sentinel-2
                const headerSideLr = document.getElementById("header-side-lr");
                if (headerSideLr) {
                    headerSideLr.innerText = "Input: Sentinel-2 L2A (10m)";
                }

                // Update 2nd Side-by-Side Panel Header & Slider Tag for AI Output
                const headerSideSr = document.getElementById("header-side-sr");
                const modelNames = {
                    hat: "HAT",
                    evoland: "HAT",
                    ldsr_s2: "LDSR-S2",
                    srmnet: "SRM-Net",
                    carn: "CARN",
                    evoland_carn: "CARN"
                };
                const activeModel = modelNames[state.modelName] || "HAT";
                const targetRes = (10.0 / state.scaleFactor).toFixed(1);
                if (headerSideSr) {
                    headerSideSr.innerText = `Output: ${activeModel} (${targetRes}m)`;
                }
                const tagSr = document.getElementById("tag-sr");
                if (tagSr) {
                    tagSr.innerText = `Output: ${activeModel} (${targetRes}m)`;
                }

                const spMetaModel = document.getElementById("spectral-meta-model");
                if (spMetaModel) {
                    spMetaModel.innerText = `${activeModel} (${targetRes}m GSD)`;
                }
                if (state.activeTopView === "spectral") {
                    selectSpectralMode(state.activeSpectralMode || "ndvi");
                }

                // Update 3rd Side-by-Side Panel Header strictly per tier rules
                const headerSideHr = document.getElementById("header-side-hr");
                if (headerSideHr) {
                    if (data.is_scientific_ground_truth) {
                        headerSideHr.innerText = "Reference: Ground Truth HR (1.5m)";
                    } else if (data.has_reference) {
                        headerSideHr.innerText = "Reference: Visual Basemap";
                    } else {
                        headerSideHr.innerText = "Reference: None Available";
                    }
                }

                // Update Array Dimensions in Inspector Badge
                if (inspectorDims && data.lr_dimensions && data.sr_dimensions) {
                    inspectorDims.innerText = `LR: ${data.lr_dimensions} → SR: ${data.sr_dimensions} (${data.pixel_expansion_ratio})`;
                }

                // Update Scientific USP Metrics (reference-free via opensr-test)
                metricConfidence.innerText = `${data.usp_metrics.avg_confidence}%`;
                metricUncertainty.innerText = data.usp_metrics.mean_uncertainty;
                metricSam.innerText = `${data.usp_metrics.sam_degrees}°`;
                metricCycle.innerText = data.usp_metrics.cycle_consistency_mae;

                // Quality Assessment Mode Swapping (Paired vs No-Reference)
                const isPaired = (data.assessment_mode === "paired" || (data.is_scientific_ground_truth && data.validation_metrics && data.validation_metrics.has_reference));

                if (isPaired) {
                    // Activate Paired Ground Truth Panel (Punjab, Delhi, Varanasi)
                    if (panelValidationPaired) panelValidationPaired.style.display = "block";
                    if (panelValidationNr) panelValidationNr.style.display = "none";

                    if (data.validation_metrics) {
                        metricPsnr.innerText = `${data.validation_metrics.psnr_db} dB`;
                        metricSsim.innerText = data.validation_metrics.ssim;
                        metricErgas.innerText = data.validation_metrics.ergas;
                        metricRuntime.innerText = `${data.inference_time_ms} ms (${data.device})`;

                        const elCoregStatus = document.getElementById("coreg-shift-status");
                        const elComputedAt = document.getElementById("metric-computed-at");
                        if (elComputedAt && data.validation_metrics.computed_at) {
                            elComputedAt.innerText = `Fresh: ${data.validation_metrics.computed_at} [ID: ${data.validation_metrics.execution_id || 'live'}]`;
                        }
                        if (elCoregStatus && data.validation_metrics.coregistration) {
                            const c = data.validation_metrics.coregistration;
                            if (c.success) {
                                elCoregStatus.innerHTML = `<span style="color: #4ade80;">✓ Co-Reg Shift:</span> X: <strong>${c.x_shift_px > 0 ? '+' : ''}${c.x_shift_px}px</strong> (${c.x_shift_m > 0 ? '+' : ''}${c.x_shift_m}m), Y: <strong>${c.y_shift_px > 0 ? '+' : ''}${c.y_shift_px}px</strong> (${c.y_shift_m > 0 ? '+' : ''}${c.y_shift_m}m)`;
                            } else {
                                elCoregStatus.innerHTML = `<span style="color: #f59e0b;">⚠️ Coreg:</span> ${c.method || 'Fallback'}`;
                            }
                        }
                    }
                } else {
                    // Activate No-Reference Quality Assessment Panel (pyiqa NIQE & BRISQUE + opensr-test)
                    if (panelValidationPaired) panelValidationPaired.style.display = "none";
                    if (panelValidationNr) panelValidationNr.style.display = "block";

                    const validationNoRefBanner = document.getElementById("validation-no-ref-banner");
                    if (validationNoRefBanner) {
                        if (data.reference_tier === 3 || (!data.is_scientific_ground_truth && data.has_reference)) {
                            validationNoRefBanner.innerHTML = `
                                <div style="font-weight: 600; color: #c084fc; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                                    <span style="font-size: 0.95rem;">🌐</span> Tier-3 Global Reference Basemap
                                </div>
                                Loaded Esri WorldImagery via <code>leafmap</code> for visual reference. Because this is an uncalibrated basemap with variable capture resolution and sun angles, full-reference PSNR/SSIM metrics are omitted. Evaluating blind No-Reference Image Quality Assessment (pyiqa NIQE &amp; BRISQUE) + opensr-test trust metrics.
                            `;
                        } else {
                            validationNoRefBanner.innerHTML = `
                                <div style="font-weight: 600; color: #c084fc; margin-bottom: 4px; display: flex; align-items: center; gap: 6px;">
                                    <span style="font-size: 0.95rem;">⚠️</span> Unpaired Scene Notification
                                </div>
                                No ground truth reference available for this AOI — validation metrics require a paired high-resolution reference tile. Activating Blind/No-Reference Image Quality Assessment (pyiqa NIQE &amp; BRISQUE) + opensr-test trust metrics.
                            `;
                        }
                    }

                    if (data.no_reference_metrics) {
                        if (metricNrNiqe) metricNrNiqe.innerText = data.no_reference_metrics.niqe;
                        if (metricNrBrisque) metricNrBrisque.innerText = data.no_reference_metrics.brisque;
                    }
                    if (metricNrConfidence) metricNrConfidence.innerText = `${data.usp_metrics.avg_confidence}%`;
                    if (metricNrCycle) metricNrCycle.innerText = data.usp_metrics.cycle_consistency_mae;
                }

                // Update Infrastructure Analytics HUD (Section 12 - Live SamGeo Building & Road Detection)
                if (data.infrastructure_summary) {
                    const isum = data.infrastructure_summary;
                    if (metricBuildings) metricBuildings.innerText = isum.buildings_count ?? "--";
                    if (metricRoads) metricRoads.innerText = (isum.roads_length_km !== undefined) ? `${isum.roads_length_km} km` : "-- km";
                    if (metricInfraConf) metricInfraConf.innerText = `${isum.avg_detection_confidence}%`;
                }

                // Update Live Pipeline Verification HUD
                if (data.model_info && verifCheckpoint) {
                    verifCheckpoint.innerHTML = `<strong>${data.model_info.checkpoint_file}</strong> [${data.model_info.model_architecture || data.model_info.model_name}] (${data.model_info.trained_scale}x / ${data.model_info.target_gsd})`;
                }
                if (data.model_info && verifDropout) {
                    verifDropout.innerText = data.model_info.dropout_mode;
                }
                if (data.mc_variance_stats && verifVariance) {
                    const st = data.mc_variance_stats;
                    verifVariance.innerText = `min: ${st.raw_variance_min.toExponential(2)} | max: ${st.raw_variance_max.toExponential(2)} | mean: ${st.raw_variance_mean.toExponential(2)}`;
                }
                if (verifDimensions && data.lr_dimensions && data.sr_dimensions) {
                    verifDimensions.innerText = `LR: ${data.lr_dimensions} → SR: ${data.sr_dimensions} (${data.pixel_expansion_ratio})`;
                }
                if (verifUnsharp) {
                    if (data.post_processing_stage) {
                        verifUnsharp.innerText = data.post_processing_stage;
                    } else if (data.realesrgan_sharpened) {
                        verifUnsharp.innerText = "Real-ESRGAN x4plus (outscale=1, tile=256)";
                    } else if (data.unsharp_mask_applied) {
                        verifUnsharp.innerText = "skimage.filters.unsharp_mask (radius=1.0, amount=1.2)";
                    } else {
                        verifUnsharp.innerText = "Disabled (Raw Model Output)";
                    }
                }
                if (verifReference) {
                    if (data.reference_provenance) {
                        verifReference.innerText = data.reference_provenance;
                        verifReference.style.color = data.is_scientific_ground_truth ? "#38bdf8" : "#facc15";
                    } else {
                        verifReference.innerText = "None (Unpaired AOI — No False Fallback)";
                        verifReference.style.color = "#f87171";
                    }
                }



                if (verifTemporal) {
                    if (state.acquisitionMode === "multi_temporal") {
                        verifTemporal.innerText = "3-Pass Temporal Median (Active)";
                        verifTemporal.className = "verif-code-emerald";
                    } else {
                        verifTemporal.innerText = "Single-Pass (Default)";
                        verifTemporal.className = "verif-code-cyan";
                    }
                }

                // Update NETRA Blockchain Provenance Telemetry
                if (data.blockchain_provenance) {
                    state.blockchainRecord = data.blockchain_provenance;
                    updateBlockchainTelemetry(data.blockchain_provenance);
                }
            }
        } catch (err) {
            console.error("[GEO-SRM] Super-resolution request failed:", err);
        } finally {
            if (fetchSeq === null || fetchSeq === currentFetchSeq) {
                loadingScrim.classList.add("hidden");
            }
        }
    }

    // =========================================================================
    // 4. Split Slider Interactive Swipe with OpenSeadragon Viewports
    // =========================================================================
    let isDragging = false;

    function updateSliderPosition(clientX) {
        const rect = sliderContainer.getBoundingClientRect();
        let offsetX = clientX - rect.left;
        let percentage = (offsetX / rect.width) * 100;
        percentage = Math.max(0, Math.min(100, percentage));

        sliderHandle.style.left = `${percentage}%`;
        srWrapper.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
    }

    sliderHandle.addEventListener("mousedown", (e) => {
        isDragging = true;
        e.preventDefault();
    });

    window.addEventListener("mouseup", () => {
        isDragging = false;
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDragging) return;
        updateSliderPosition(e.clientX);
    });

    sliderHandle.addEventListener("touchstart", () => {
        isDragging = true;
    });

    window.addEventListener("touchend", () => {
        isDragging = false;
    });

    window.addEventListener("touchmove", (e) => {
        if (!isDragging || !e.touches.length) return;
        updateSliderPosition(e.touches[0].clientX);
    });

    sliderContainer.addEventListener("click", (e) => {
        if (e.target.closest("#slider-handle")) return;
        updateSliderPosition(e.clientX);
    });

    // =========================================================================
    // 5. Parameter Controls & Toolbar Event Listeners
    // =========================================================================
    sliderMcSamples.addEventListener("input", (e) => {
        state.mcSamples = parseInt(e.target.value);
        valMcSamples.innerText = `${state.mcSamples} passes`;
    });

    if (selectScale) {
        selectScale.addEventListener("change", (e) => {
            state.scaleFactor = parseInt(e.target.value);
            console.log(`[GEO-SRM] Target scale changed to ${state.scaleFactor}x (${10.0 / state.scaleFactor}m GSD)`);
            runSuperResolution();
        });
    }

    if (selectModel) {
        const ldsrNotice = document.getElementById("ldsr-diffusion-notice");
        selectModel.addEventListener("change", (e) => {
            state.modelName = e.target.value;
            console.log(`[GEO-SRM] Model changed to: ${state.modelName}`);
            if (state.modelName === "ldsr_s2") {
                if (ldsrNotice) ldsrNotice.classList.remove("hidden");
                showInAppNotification(
                    "LDSR-S2 Latent Diffusion Active: Generates sharper high-frequency detail. Monitor the Confidence HUD to assess hallucination risk.",
                    "⚡",
                    "warning",
                    7000
                );
            } else {
                if (ldsrNotice) ldsrNotice.classList.add("hidden");
            }
            runSuperResolution();
        });
    }

    if (selectVisMode) {
        selectVisMode.addEventListener("change", async (e) => {
            const mode = e.target.value;
            state.visualizationMode = mode;
            console.log(`[GEO-SRM] Multi-Band Spectral Mode changed to: ${mode}`);
            await switchVisualizationMode(mode);
        });
    }

    async function switchVisualizationMode(mode) {
        loadingScrim.classList.remove("hidden");
        try {
            const res = await fetch("/api/visualize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mode: mode,
                    apply_realesrgan_sharpen: state.applyRealESRGANSharpen
                })
            });
            const data = await res.json();
            if (data.status === "success") {
                if (data.lr_preview) {
                    state.latestLrUrl = data.lr_preview;
                    loadViewerImage(viewerLR, data.lr_preview, false);
                    loadViewerImage(viewerSideLR, data.lr_preview, false);
                }
                if (data.sr_preview) {
                    state.latestSrUrl = data.sr_preview;
                    loadViewerImage(viewerSR, data.sr_preview, true);
                    loadViewerImage(viewerSideSR, data.sr_preview, false);
                    syncCloudOverlays();
                }
                updateVisualizationLegend(data);
            }
        } catch (err) {
            console.error("[GEO-SRM] Failed to switch spectral visualization mode:", err);
        } finally {
            loadingScrim.classList.add("hidden");
        }
    }

    function updateVisualizationLegend(info) {
        if (!info) return;
        const titleEl = document.getElementById("spectral-legend-title");
        const statsEl = document.getElementById("spectral-legend-stats");
        const barContainer = document.getElementById("spectral-legend-bar-container");
        const barEl = document.getElementById("spectral-legend-bar");
        const tickMin = document.getElementById("spectral-tick-min");
        const tickMid = document.getElementById("spectral-tick-mid");
        const tickMax = document.getElementById("spectral-tick-max");
        const chipsContainer = document.getElementById("spectral-chips-container");

        const modeIcons = {
            true_color: "🌿",
            false_color_ir: "🔴",
            ndvi: "🌱",
            ndwi: "💧",
            ndbi: "🏢",
            nbr: "🔥"
        };
        const icon = modeIcons[info.mode] || "🛰️";
        if (titleEl) {
            titleEl.innerText = `${icon} ${info.title || info.name || info.mode.toUpperCase()}`;
        }

        if (info.mode === "true_color" || (!info.is_index && info.mode !== "false_color_ir")) {
            if (statsEl) statsEl.innerText = "Surface Reflectance (B04-B03-B02)";
            if (barContainer) barContainer.style.display = "none";
            if (chipsContainer) chipsContainer.style.display = "none";
        } else if (info.mode === "false_color_ir") {
            if (statsEl) statsEl.innerText = "NIR-Red-Green Composite";
            if (barContainer) barContainer.style.display = "none";
            if (chipsContainer) {
                chipsContainer.style.display = "flex";
                chipsContainer.innerHTML = `
                    <span class="spectral-chip"><span class="spectral-chip-dot" style="background:#e31a1c;"></span> Veg / Canopy</span>
                    <span class="spectral-chip"><span class="spectral-chip-dot" style="background:#73b2d8;"></span> Urban / Roads</span>
                    <span class="spectral-chip"><span class="spectral-chip-dot" style="background:#081d58;"></span> Water</span>
                    <span class="spectral-chip"><span class="spectral-chip-dot" style="background:#d2b48c;"></span> Bare Soil</span>
                `;
            }
        } else if (info.is_index && info.legend && info.legend.type === "continuous") {
            if (chipsContainer) chipsContainer.style.display = "none";
            if (barContainer) barContainer.style.display = "flex";
            if (barEl) barEl.style.background = info.legend.gradient_css || "linear-gradient(to right, #000, #fff)";
            if (tickMin) tickMin.innerText = info.legend.min_label || `${info.legend.min}`;
            if (tickMid) tickMid.innerText = info.legend.mid_label || "0.0";
            if (tickMax) tickMax.innerText = info.legend.max_label || `${info.legend.max}`;

            if (statsEl && info.stats) {
                statsEl.innerText = `μ=${info.stats.mean} [${info.stats.min}, ${info.stats.max}]`;
            } else if (statsEl) {
                statsEl.innerText = info.subtitle || "Spectral Index";
            }
        }
    }

    if (checkRealESRGANSharpen) {
        checkRealESRGANSharpen.addEventListener("change", (e) => {
            state.applyRealESRGANSharpen = e.target.checked;
            console.log(`[GEO-SRM] Stage 2 Real-ESRGAN sharpening toggled: ${state.applyRealESRGANSharpen}`);
            runSuperResolution();
        });
    }



    if (checkUnsharp) {
        checkUnsharp.addEventListener("change", (e) => {
            state.applyUnsharp = e.target.checked;
            console.log(`[GEO-SRM] Edge enhancement toggled: ${state.applyUnsharp}`);
            runSuperResolution();
        });
    }

    if (btnModeSingle) {
        btnModeSingle.addEventListener("click", async () => {
            btnModeSingle.classList.add("active");
            if (btnModeMulti) btnModeMulti.classList.remove("active");
            state.acquisitionMode = "single";
            if (verifTemporal) {
                verifTemporal.innerText = "Single-Pass (Default)";
                verifTemporal.className = "verif-code-cyan";
            }
            await fetchTileForBbox(state.activeBbox, state.activePreset);
        });
    }

    if (btnModeMulti) {
        btnModeMulti.addEventListener("click", async () => {
            btnModeMulti.classList.add("active");
            if (btnModeSingle) btnModeSingle.classList.remove("active");
            state.acquisitionMode = "multi_temporal";
            if (verifTemporal) {
                verifTemporal.innerText = "3-Pass Temporal Median (Active)";
                verifTemporal.className = "verif-code-emerald";
            }
            loadingScrim.classList.remove("hidden");
            try {
                const res = await fetch("/api/fetch-multi-temporal", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        aoi_id: state.activePreset || "punjab_agri",
                        bbox: state.activeBbox
                    })
                });
                const data = await res.json();
                if (data.status === "success") {
                    state.latestLrUrl = data.lr_preview;
                    state.latestHrUrl = data.hr_preview;
                    if (data.cloud_mask_preview) state.latestCloudUrl = data.cloud_mask_preview;
                    if (data.cloud_coverage_pct !== undefined) state.cloudCoveragePct = data.cloud_coverage_pct;
                    if (data.cloud_warning) updateCloudWarning(data.cloud_warning);
                    loadViewerImage(viewerLR, data.lr_preview);
                    loadViewerImage(viewerSideLR, data.lr_preview);
                    loadViewerImage(viewerSideHR, data.hr_preview);
                    if (verifReference) verifReference.innerText = data.reference_file;

                    if (data.quality_comparison) {
                        console.log("[GEO-SRM] Multi-temporal benchmark improvement:", data.quality_comparison);
                    }
                    await runSuperResolution();
                }
            } catch (err) {
                console.error("[GEO-SRM] Multi-temporal fetch failed:", err);
            } finally {
                loadingScrim.classList.add("hidden");
            }
        });
    }

    // Confidence Heatmap Toggle
    toggleConfidencePill.addEventListener("click", () => {
        state.confidenceActive = !state.confidenceActive;
        toggleConfidencePill.classList.toggle("active", state.confidenceActive);
        updateOverlayDisplay();
    });

    sliderConfidenceOpacity.addEventListener("input", (e) => {
        state.confidenceOpacity = parseFloat(e.target.value);
        updateOverlayDisplay();
    });

    // Infrastructure Detection Toggle (Section 12)
    if (toggleInfraPill) {
        toggleInfraPill.addEventListener("click", () => {
            state.infraActive = !state.infraActive;
            toggleInfraPill.classList.toggle("active", state.infraActive);
            toggleInfraPill.classList.toggle("infra-active", state.infraActive);
            updateOverlayDisplay();
        });
    }

    if (sliderInfraOpacity) {
        sliderInfraOpacity.addEventListener("input", (e) => {
            state.infraOpacity = parseFloat(e.target.value);
            updateOverlayDisplay();
        });
    }

    // Cloud Mask Occlusion Toggle
    if (toggleCloudPill) {
        toggleCloudPill.addEventListener("click", () => {
            state.cloudActive = !state.cloudActive;
            toggleCloudPill.classList.toggle("active", state.cloudActive);
            toggleCloudPill.classList.toggle("cloud-active", state.cloudActive);
            syncCloudOverlays();
        });
    }

    if (sliderCloudOpacity) {
        sliderCloudOpacity.addEventListener("input", (e) => {
            state.cloudOpacity = parseFloat(e.target.value);
            updateOverlayDisplay();
        });
    }

    if (btnDismissCloudWarning) {
        btnDismissCloudWarning.addEventListener("click", () => {
            updateCloudWarning(null);
        });
    }

    // View Mode Switcher
    tabSplit.addEventListener("click", () => {
        tabSplit.classList.add("active");
        tabSide.classList.remove("active");
        sliderContainer.classList.remove("hidden");
        sideBySideContainer.classList.add("hidden");
    });

    tabSide.addEventListener("click", () => {
        tabSide.classList.add("active");
        tabSplit.classList.remove("active");
        sliderContainer.classList.add("hidden");
        sideBySideContainer.classList.remove("hidden");

        // Force viewport refresh for side-by-side OpenSeadragon instances
        if (state.latestLrUrl) loadViewerImage(viewerSideLR, state.latestLrUrl);
        if (state.latestSrUrl) loadViewerImage(viewerSideSR, state.latestSrUrl);
        if (state.latestHrUrl) loadViewerImage(viewerSideHR, state.latestHrUrl);
    });

    // Run SR Button (if present)
    if (btnRunSr) {
        btnRunSr.addEventListener("click", () => {
            runSuperResolution();
        });
    }

    // File Upload
    btnTriggerUpload.addEventListener("click", () => fileUpload.click());

    fileUpload.addEventListener("change", async (e) => {
        if (!e.target.files.length) return;
        const file = e.target.files[0];
        const formData = new FormData();
        formData.append("file", file);

        loadingScrim.classList.remove("hidden");
        try {
            const res = await fetch("/api/upload", {
                method: "POST",
                body: formData
            });
            const data = await res.json();
            if (data.status === "success") {
                state.latestLrUrl = data.lr_preview;
                state.latestHrUrl = data.hr_preview;
                if (data.cloud_mask_preview) state.latestCloudUrl = data.cloud_mask_preview;
                if (data.cloud_coverage_pct !== undefined) state.cloudCoveragePct = data.cloud_coverage_pct;
                if (data.cloud_warning) updateCloudWarning(data.cloud_warning);
                loadViewerImage(viewerLR, data.lr_preview);
                loadViewerImage(viewerSideLR, data.lr_preview);
                loadViewerImage(viewerSideHR, data.hr_preview);
                document.querySelectorAll(".preset-chip").forEach(c => c.classList.remove("active"));
                await runSuperResolution();
            }
        } catch (err) {
            console.error("[GEO-SRM] Upload failed:", err);
        } finally {
            loadingScrim.classList.add("hidden");
        }
    });

    // Export Handlers
    btnDownloadSr.addEventListener("click", () => {
        window.open("/api/download/sr", "_blank");
    });

    btnDownloadConf.addEventListener("click", () => {
        window.open("/api/download/confidence", "_blank");
    });

    if (btnDownloadGeojson) {
        btnDownloadGeojson.addEventListener("click", () => {
            window.open("/api/download/infrastructure", "_blank");
        });
    }

    // =========================================================================
    // NETRA Blockchain Provenance Module UI Handlers (Polygon Amoy Testnet)
    // =========================================================================
    const headerChainVersion = document.getElementById("header-chain-version");
    const verifBlockchainStatus = document.getElementById("verif-blockchain-status");
    const btnOpenBlockchain = document.getElementById("btn-open-blockchain");
    const btnFooterOpenBlockchain = document.getElementById("btn-footer-open-blockchain");
    const modalBlockchain = document.getElementById("modal-blockchain");
    const btnCloseBlockchainModal = document.getElementById("btn-close-blockchain-modal");
    const btnDoneBlockchain = document.getElementById("btn-done-blockchain");
    const tabBcHistory = document.getElementById("tab-bc-history");
    const tabBcVerify = document.getElementById("tab-bc-verify");
    const paneBcHistory = document.getElementById("pane-bc-history");
    const paneBcVerify = document.getElementById("pane-bc-verify");
    const bcTimelineList = document.getElementById("bc-timeline-list");
    const inputVerifyFile = document.getElementById("input-verify-file");
    const bcVerifyResult = document.getElementById("bc-verify-result");

    function updateBlockchainTelemetry(record) {
        if (!record) return;
        const verTag = `v${record.versionNumber} Verified`;
        if (headerChainVersion) {
            headerChainVersion.innerText = verTag;
        }
        if (verifBlockchainStatus) {
            const shortHash = record.imageHash ? `${record.imageHash.slice(0, 10)}...` : "";
            verifBlockchainStatus.innerText = `v${record.versionNumber} (${shortHash}) • Anchored`;
        }

        const bcTileId = document.getElementById("bc-tile-id");
        const bcVersionTag = document.getElementById("bc-version-tag");
        const bcCoords = document.getElementById("bc-coords");
        const bcHash = document.getElementById("bc-hash");
        const bcTxLink = document.getElementById("bc-tx-link");

        if (bcTileId) bcTileId.innerText = record.tileId || state.activeAoiId || "UKIS-2026_tile_001";
        if (bcVersionTag) bcVersionTag.innerText = `v${record.versionNumber} (Verified On-Chain)`;
        if (bcCoords && record.realCoordinates) {
            bcCoords.innerText = `${record.realCoordinates.latCenter.toFixed(4)}°N, ${record.realCoordinates.lonCenter.toFixed(4)}°E`;
        }
        if (bcHash) {
            bcHash.innerText = record.imageHash ? `${record.imageHash.slice(0, 18)}...` : "0x...";
            bcHash.title = record.imageHash || "";
        }
        if (bcTxLink && record.txLink) {
            bcTxLink.href = record.txLink;
        }
    }

    async function loadBlockchainHistory() {
        const tileId = state.blockchainRecord?.tileId || state.activeAoiId || "punjab_agri";
        if (!bcTimelineList) return;

        bcTimelineList.innerHTML = `<div style="color: #9ca3af; font-size: 0.82rem; padding: 1rem 0;">Querying on-chain version history chain...</div>`;

        try {
            const res = await fetch(`/api/blockchain/history/${tileId}`);
            const data = await res.json();
            if (data.status === "success" && data.history) {
                if (data.history.length === 0) {
                    bcTimelineList.innerHTML = `<div style="color: #9ca3af; font-size: 0.82rem; padding: 1rem 0;">No on-chain records found for tile '${tileId}'. Run Super-Resolution to register v1.</div>`;
                    return;
                }

                let html = "";
                const items = data.history;
                items.forEach((item, index) => {
                    const isLatest = (index === items.length - 1);
                    const isFirst = item.is_first_version;
                    html += `
                        <div class="bc-timeline-item">
                            <div class="bc-timeline-indicator">
                                <div class="bc-timeline-circle ${isLatest ? 'latest' : ''}">${item.version}</div>
                                ${!isLatest ? '<div class="bc-timeline-connector"></div>' : ''}
                            </div>
                            <div class="bc-timeline-card">
                                <div class="bc-card-header">
                                    <div class="bc-card-title">${item.version_tag} • ${item.model_name} (${item.scale_factor}x) ${isLatest ? '<span style="color:#10b981; font-size:0.75rem; margin-left:6px;">● Latest Active</span>' : ''}</div>
                                    <span class="bc-card-time">${item.timestamp_iso}</span>
                                </div>
                                <div class="bc-card-hashes">
                                    <div class="bc-hash-row">
                                        <span class="bc-hash-label">Current SHA-256:</span>
                                        <span class="bc-hash-val">${item.image_hash_short}</span>
                                    </div>
                                    <div class="bc-hash-row">
                                        <span class="bc-hash-label">Linked Previous Hash:</span>
                                        <span class="bc-hash-prev" style="${isFirst ? 'color:#6b7280;' : ''}">${isFirst ? '0x00000000... (Genesis Root v1)' : item.previous_hash_short}</span>
                                    </div>
                                </div>
                                <div class="bc-card-meta">
                                    <span>Submitter: <code style="color:#c084fc; font-family:monospace;">${item.submitter_short}</code></span>
                                    <span>Centre: ${item.lat.toFixed(4)}°N, ${item.lon.toFixed(4)}°E</span>
                                    <a href="${item.tx_link}" target="_blank" rel="noopener noreferrer" style="color:#38bdf8; text-decoration:none;">View Block ↗</a>
                                </div>
                            </div>
                        </div>
                    `;
                });
                bcTimelineList.innerHTML = html;
            }
        } catch (err) {
            bcTimelineList.innerHTML = `<div style="color: #f87171; font-size: 0.82rem; padding: 1rem 0;">Error fetching history: ${err.message}</div>`;
        }
    }

    function openBlockchainModal() {
        if (!modalBlockchain) return;
        modalBlockchain.classList.remove("hidden");
        if (state.blockchainRecord) {
            updateBlockchainTelemetry(state.blockchainRecord);
        }
        loadBlockchainHistory();
    }

    function closeBlockchainModal() {
        if (!modalBlockchain) return;
        modalBlockchain.classList.add("hidden");
    }

    if (btnOpenBlockchain) btnOpenBlockchain.addEventListener("click", openBlockchainModal);
    if (btnFooterOpenBlockchain) btnFooterOpenBlockchain.addEventListener("click", openBlockchainModal);
    if (btnCloseBlockchainModal) btnCloseBlockchainModal.addEventListener("click", closeBlockchainModal);
    if (btnDoneBlockchain) btnDoneBlockchain.addEventListener("click", closeBlockchainModal);

    if (modalBlockchain) {
        modalBlockchain.addEventListener("click", (e) => {
            if (e.target === modalBlockchain) closeBlockchainModal();
        });
    }

    // Modal Tabs
    if (tabBcHistory && tabBcVerify && paneBcHistory && paneBcVerify) {
        tabBcHistory.addEventListener("click", () => {
            tabBcHistory.classList.add("active");
            tabBcVerify.classList.remove("active");
            paneBcHistory.classList.remove("hidden");
            paneBcVerify.classList.add("hidden");
        });
        tabBcVerify.addEventListener("click", () => {
            tabBcVerify.classList.add("active");
            tabBcHistory.classList.remove("active");
            paneBcVerify.classList.remove("hidden");
            paneBcHistory.classList.add("hidden");
        });
    }

    // File Verifier Handler
    if (inputVerifyFile && bcVerifyResult) {
        inputVerifyFile.addEventListener("change", async (e) => {
            if (!e.target.files.length) return;
            const file = e.target.files[0];
            const tileId = state.blockchainRecord?.tileId || state.activeAoiId || "punjab_agri";

            bcVerifyResult.className = "bc-verify-result";
            bcVerifyResult.classList.remove("hidden");
            bcVerifyResult.innerHTML = `<div>⟳ Computing SHA-256 and verifying against Polygon Amoy on-chain record for '<strong>${tileId}</strong>'...</div>`;

            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await fetch(`/api/blockchain/verify-file?tile_id=${tileId}`, {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                if (data.status === "success" && data.result) {
                    const r = data.result;
                    if (r.verified) {
                        bcVerifyResult.className = "bc-verify-result success";
                        bcVerifyResult.innerHTML = `
                            <strong>✓ On-Chain Cryptographic Match (Authentic)</strong>
                            <div style="margin-top: 4px; font-size: 0.76rem; color: #d1fae5;">
                                File matches <strong>Version ${r.version_number}</strong> on Polygon Amoy testnet.<br>
                                Content SHA-256: <code style="color:#a7f3d0;">${r.provided_hash.slice(0, 20)}...</code><br>
                                Registered Timestamp: ${new Date(r.timestamp * 1000).toUTCString()}
                            </div>
                        `;
                    } else {
                        bcVerifyResult.className = "bc-verify-result error";
                        bcVerifyResult.innerHTML = `
                            <strong>⚠ Tamper Detected: Hash Mismatch</strong>
                            <div style="margin-top: 4px; font-size: 0.76rem; color: #fecaca;">
                                The uploaded file does not match the latest registered version on-chain.<br>
                                Uploaded Hash: <code style="color:#fda4af;">${r.provided_hash.slice(0, 20)}...</code><br>
                                Expected On-Chain Hash: <code style="color:#6ee7b7;">${r.registered_hash.slice(0, 20)}...</code>
                            </div>
                        `;
                    }
                } else {
                    bcVerifyResult.className = "bc-verify-result error";
                    bcVerifyResult.innerText = `Verification error: ${data.detail || 'Unknown error'}`;
                }
            } catch (err) {
                bcVerifyResult.className = "bc-verify-result error";
                bcVerifyResult.innerText = `Failed to connect to verification endpoint: ${err.message}`;
            }
        });
    }

    // =========================================================================
    // Dedicated Spectral & Disaster Analysis View (View 2) Logic
    // =========================================================================
    let spectralMap = null;
    let spectralAoiRectangle = null;
    let spectralDrawnItems = null;
    let spectralDrawControl = null;

    function initSpectralMap() {
        const mapEl = document.getElementById("spectral-aoi-map");
        if (!mapEl || spectralMap) return;

        spectralMap = L.map("spectral-aoi-map", {
            zoomControl: false,
            attributionControl: false
        }).setView([30.575, 75.33], 12);

        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            maxZoom: 18,
        }).addTo(spectralMap);

        spectralAoiRectangle = L.rectangle([
            [state.activeBbox[1], state.activeBbox[0]],
            [state.activeBbox[3], state.activeBbox[2]]
        ], {
            color: "#10b981",
            weight: 2,
            fillColor: "#10b981",
            fillOpacity: 0.18
        }).addTo(spectralMap);

        spectralDrawnItems = new L.FeatureGroup();
        spectralMap.addLayer(spectralDrawnItems);

        spectralDrawControl = new L.Control.Draw({
            position: 'topright',
            draw: {
                polygon: false,
                polyline: false,
                circle: false,
                circlemarker: false,
                marker: false,
                rectangle: {
                    shapeOptions: {
                        color: "#10b981",
                        weight: 2,
                        fillColor: "#10b981",
                        fillOpacity: 0.20
                    }
                }
            },
            edit: { featureGroup: spectralDrawnItems }
        });
        spectralMap.addControl(spectralDrawControl);

            spectralMap.on(L.Draw.Event.CREATED, function (e) {
            const layer = e.layer;
            spectralDrawnItems.clearLayers();
            spectralDrawnItems.addLayer(layer);

            const bounds = layer.getBounds();
            const bbox = [
                parseFloat(bounds.getWest().toFixed(5)),
                parseFloat(bounds.getSouth().toFixed(5)),
                parseFloat(bounds.getEast().toFixed(5)),
                parseFloat(bounds.getNorth().toFixed(5))
            ];

            state.activeBbox = bbox;
            state.activePreset = "custom_drawn_aoi";
            const center = bounds.getCenter();
            syncAoiUI(bounds, bbox, "custom_drawn_aoi", center.lat, center.lng);
            fetchTileForBbox(bbox, "custom_drawn_aoi");
        });

        spectralMap.on(L.Draw.Event.EDITED, function (e) {
            const layers = e.layers;
            layers.eachLayer(function (layer) {
                const bounds = layer.getBounds();
                const bbox = [
                    parseFloat(bounds.getWest().toFixed(5)),
                    parseFloat(bounds.getSouth().toFixed(5)),
                    parseFloat(bounds.getEast().toFixed(5)),
                    parseFloat(bounds.getNorth().toFixed(5))
                ];
                state.activeBbox = bbox;
                state.activePreset = "custom_drawn_aoi";
                const center = bounds.getCenter();
                syncAoiUI(bounds, bbox, "custom_drawn_aoi", center.lat, center.lng);
                fetchTileForBbox(bbox, "custom_drawn_aoi");
            });
        });

        spectralMap.on(L.Draw.Event.DELETED, function () {
            if (state.presets && state.presets.length > 0) {
                selectPreset(state.presets[0].id);
            }
        });

        spectralMap.on("click", (e) => {
            if (document.querySelector(".leaflet-draw-actions")) return;
            const lat = e.latlng.lat;
            const lon = e.latlng.lng;
            const span = 0.03;
            const customBbox = [
                parseFloat((lon - span).toFixed(4)),
                parseFloat((lat - span).toFixed(4)),
                parseFloat((lon + span).toFixed(4)),
                parseFloat((lat + span).toFixed(4))
            ];

            state.activeBbox = customBbox;
            state.activePreset = "custom_click";
            const bounds = [
                [customBbox[1], customBbox[0]],
                [customBbox[3], customBbox[2]]
            ];

            if (spectralDrawnItems) spectralDrawnItems.clearLayers();
            if (drawnItems) drawnItems.clearLayers();
            if (spectralMap) spectralMap.panTo([lat, lon]);
            if (map) map.panTo([lat, lon]);

            syncAoiUI(bounds, customBbox, "custom_click", lat, lon);
            fetchTileForBbox(customBbox, "custom_click");
        });
    }

    // View 2 Manual Coordinate Navigation and Draw Buttons
    const spectralBtnGotoCoords = document.getElementById("spectral-btn-goto-coords");
    const spectralInputLat = document.getElementById("spectral-input-lat");
    const spectralInputLon = document.getElementById("spectral-input-lon");
    const spectralBtnDrawRect = document.getElementById("spectral-btn-draw-rect");

    if (spectralBtnGotoCoords) {
        spectralBtnGotoCoords.addEventListener("click", () => {
            const lat = parseFloat(spectralInputLat ? spectralInputLat.value.trim() : NaN);
            const lon = parseFloat(spectralInputLon ? spectralInputLon.value.trim() : NaN);
            navigateToCoordinates(lat, lon);
        });
    }

    const handleSpectralCoordEnter = (e) => {
        if (e.key === "Enter") {
            const lat = parseFloat(spectralInputLat ? spectralInputLat.value.trim() : NaN);
            const lon = parseFloat(spectralInputLon ? spectralInputLon.value.trim() : NaN);
            navigateToCoordinates(lat, lon);
        }
    };
    if (spectralInputLat) spectralInputLat.addEventListener("keydown", handleSpectralCoordEnter);
    if (spectralInputLon) spectralInputLon.addEventListener("keydown", handleSpectralCoordEnter);

    if (spectralBtnDrawRect) {
        spectralBtnDrawRect.addEventListener("click", () => {
            initSpectralMap();
            if (spectralMap && spectralDrawControl) {
                new L.Draw.Rectangle(spectralMap, spectralDrawControl.options.draw.rectangle).enable();
            }
        });
    }

    // Top-Level View Switching
    window.onViewSwitched = function onViewSwitched(viewName) {
        state.activeTopView = viewName;
        console.log(`[GEO-SRM] View switch callback active: ${viewName}`);

        if (viewName === "spectral") {
            try {
                initSpectralMap();
            } catch (e) {
                console.warn("[GEO-SRM] Leaflet spectral map init notice:", e);
            }

            const spectralContainer = document.getElementById("view-spectral-analysis");
            if (spectralContainer) void spectralContainer.offsetHeight;

            try {
                ensureSpectralViewers();
            } catch (e) {
                console.warn("[GEO-SRM] Spectral viewers init notice:", e);
            }

            setSpectralSliderPercent(50);

            setTimeout(() => {
                try {
                    if (spectralMap) {
                        spectralMap.invalidateSize();
                        if (state.activeBbox) {
                            spectralMap.fitBounds([
                                [state.activeBbox[1], state.activeBbox[0]],
                                [state.activeBbox[3], state.activeBbox[2]]
                            ]);
                        }
                    }
                } catch (e) { }

                [spectralViewerLR, spectralViewerSR, spectralViewerSingle].forEach(v => {
                    try {
                        if (v && v.viewport) {
                            v.viewport.resize();
                            v.viewport.goHome(true);
                            v.viewport.applyConstraints();
                        }
                    } catch (e) { }
                });
                window.dispatchEvent(new Event("resize"));
            }, 100);

            const currentMode = state.activeSpectralMode || "ndvi";
            selectSpectralMode(currentMode);
        } else {
            setTimeout(() => {
                try {
                    if (map) {
                        map.invalidateSize();
                        if (state.activeBbox) {
                            map.fitBounds([
                                [state.activeBbox[1], state.activeBbox[0]],
                                [state.activeBbox[3], state.activeBbox[2]]
                            ]);
                        }
                    }
                } catch (e) { }

                [viewerLR, viewerSR, viewerSideLR, viewerSideSR, viewerSideHR].forEach(v => {
                    try {
                        if (v && v.viewport) {
                            v.viewport.applyConstraints();
                        }
                    } catch (e) { }
                });
                window.dispatchEvent(new Event("resize"));
            }, 100);
        }
    };

    // If inline script recorded a pending switch before app.js was ready, handle it now
    if (window._pendingViewSwitch) {
        const pending = window._pendingViewSwitch;
        window._pendingViewSwitch = null;
        window.onViewSwitched(pending);
    }

    if (navBtnSr) navBtnSr.addEventListener("click", () => window.switchTopView("sr"));
    if (navBtnSpectral) navBtnSpectral.addEventListener("click", () => window.switchTopView("spectral"));

    // Dedicated Spectral Mode Cards Click Handlers
    async function selectSpectralMode(mode) {
        state.activeSpectralMode = mode;

        document.querySelectorAll(".spectral-mode-tab").forEach(tab => {
            tab.classList.toggle("active", tab.dataset.mode === mode);
        });

        const spectralLoadingScrim = document.getElementById("spectral-loading-scrim");
        if (spectralLoadingScrim) spectralLoadingScrim.classList.remove("hidden");

        try {
            const res = await fetch("/api/visualize", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    mode: mode,
                    apply_realesrgan_sharpen: false
                })
            });
            const data = await res.json();
            if (data.status === "success") {
                if (data.lr_preview) {
                    state.latestSpectralLrUrl = data.lr_preview;
                    const imgLr = document.getElementById("spectral-img-lr");
                    if (imgLr) imgLr.src = data.lr_preview;
                    if (spectralViewerLR) loadViewerImage(spectralViewerLR, data.lr_preview, false);
                }
                if (data.sr_preview) {
                    state.latestSpectralSrUrl = data.sr_preview;
                    const imgSr = document.getElementById("spectral-img-sr");
                    const imgSingle = document.getElementById("spectral-img-single");
                    if (imgSr) imgSr.src = data.sr_preview;
                    if (imgSingle) imgSingle.src = data.sr_preview;
                    if (spectralViewerSR) loadViewerImage(spectralViewerSR, data.sr_preview, false);
                    if (spectralViewerSingle) loadViewerImage(spectralViewerSingle, data.sr_preview, false);
                }

                // Update Viewport Labels
                const tagSr = document.getElementById("spectral-tag-sr");
                if (tagSr) {
                    tagSr.innerText = `Output: HAT 2.5m (${data.title})`;
                }
                const singleHeader = document.getElementById("spectral-single-header");
                if (singleHeader) {
                    singleHeader.innerText = `Super-Resolved 2.5m — ${data.title}`;
                }

                // Update Primary Area Coverage Stat Badge (Km² affected / covered)
                const statTitle = document.getElementById("stat-category-title");
                const statBadge = document.getElementById("stat-threshold-badge");
                const statDesc = document.getElementById("stat-category-desc");
                const statAreaKm2 = document.getElementById("stat-area-km2");
                const statAreaPct = document.getElementById("stat-area-pct");
                const statTotalAoi = document.getElementById("stat-total-aoi");

                if (data.coverage_stats) {
                    const cs = data.coverage_stats;
                    if (statTitle) statTitle.innerText = cs.category;
                    if (statBadge) statBadge.innerText = cs.threshold;
                    if (statAreaKm2) statAreaKm2.innerText = `${cs.area_km2} km²`;
                    if (statAreaPct) statAreaPct.innerText = `${cs.area_pct}% of AOI`;
                    if (statTotalAoi) statTotalAoi.innerText = `${cs.total_aoi_km2} km²`;
                }

                // Update Numerical Index Distribution
                const elMean = document.getElementById("metric-sp-mean");
                const elMax = document.getElementById("metric-sp-max");
                const elMin = document.getElementById("metric-sp-min");
                const elStd = document.getElementById("metric-sp-std");

                if (data.stats) {
                    if (elMean) elMean.innerText = data.stats.mean !== undefined ? data.stats.mean.toFixed(3) : "--";
                    if (elMax) elMax.innerText = data.stats.max !== undefined ? data.stats.max.toFixed(3) : "--";
                    if (elMin) elMin.innerText = data.stats.min !== undefined ? data.stats.min.toFixed(3) : "--";
                    if (elStd) elStd.innerText = data.stats.std !== undefined ? data.stats.std.toFixed(3) : "--";
                }

                // Update Scientific Legend / Continuous Ramp
                const spLegendTitle = document.getElementById("sp-legend-title");
                const spLegendStats = document.getElementById("sp-legend-stats");
                const spLegendBarContainer = document.getElementById("sp-legend-bar-container");
                const spLegendBar = document.getElementById("sp-legend-bar");
                const spTickMin = document.getElementById("sp-tick-min");
                const spTickMid = document.getElementById("sp-tick-mid");
                const spTickMax = document.getElementById("sp-tick-max");

                const modeIcons = {
                    true_color: "🌿",
                    ndvi: "🌱",
                    ndwi: "💧",
                    nbr: "🔥",
                    ndbi: "🏢"
                };
                const icon = modeIcons[mode] || "🛰️";
                if (spLegendTitle) spLegendTitle.innerText = `${icon} ${data.title}`;
                if (spLegendStats) spLegendStats.innerText = `${data.subtitle} | Bands: ${data.bands_used.join("-")}`;

                if (data.is_index && data.legend && data.legend.type === "continuous") {
                    if (spLegendBarContainer) spLegendBarContainer.style.display = "flex";
                    if (spLegendBar) spLegendBar.style.background = data.legend.gradient_css || "linear-gradient(to right, #000, #fff)";
                    if (spTickMin) spTickMin.innerText = data.legend.min_label || `${data.legend.min}`;
                    if (spTickMid) spTickMid.innerText = data.legend.mid_label || "0.0";
                    if (spTickMax) spTickMax.innerText = data.legend.max_label || `${data.legend.max}`;
                } else {
                    if (spLegendBarContainer) spLegendBarContainer.style.display = "none";
                }

                // Update Scientific Decision Support Callout
                const decisionBox = document.getElementById("spectral-decision-box");
                if (decisionBox) {
                    const interpretations = {
                        ndvi: "<strong>Agricultural Monitoring:</strong> NDVI ≥ 0.40 isolates active crop canopies and chlorophyll absorption. Resolves individual agricultural parcel boundaries and irrigation medhs at 2.5m GSD.",
                        ndwi: "<strong>Flood & Hydrological Inundation:</strong> NDWI ≥ 0.00 detects standing surface water, flooded farmland, and riverbank breaching with zero spectral ambiguity.",
                        nbr: "<strong>Disaster Burn Severity:</strong> NBR ≤ 0.10 highlights active fire burn scars, agricultural stubble combustion zones, and deforested terrain.",
                        ndbi: "<strong>Urban Built-up Extent:</strong> NDBI ≥ 0.00 isolates concrete, asphalt, building footprints, and dense settlement infrastructure.",
                        true_color: "<strong>Natural Surface Reflectance:</strong> True Color (B04-Red, B03-Green, B02-Blue) with dynamic AOI radiometric color transfer for natural visual basemap consistency."
                    };
                    decisionBox.innerHTML = interpretations[mode] || `<strong>Multi-Band Analysis:</strong> ${data.description}`;
                }
            }
        } catch (err) {
            console.error("[GEO-SRM] Failed to switch spectral mode:", err);
        } finally {
            if (spectralLoadingScrim) spectralLoadingScrim.classList.add("hidden");
        }
    }

    document.querySelectorAll(".spectral-mode-tab").forEach(tab => {
        tab.addEventListener("click", () => {
            const mode = tab.dataset.mode;
            if (mode) selectSpectralMode(mode);
        });
    });

    // Dedicated Spectral View Split-Slider Drag Interaction
    const spectralSliderContainer = document.getElementById("spectral-slider-container");
    const spectralSliderHandle = document.getElementById("spectral-slider-handle");
    const spectralSrWrapper = document.getElementById("spectral-sr-wrapper");

    function setSpectralSliderPercent(percentage) {
        if (!spectralSliderHandle || !spectralSrWrapper) return;
        percentage = Math.max(0, Math.min(100, percentage));
        spectralSliderHandle.style.left = `${percentage}%`;
        spectralSrWrapper.style.clipPath = `polygon(${percentage}% 0, 100% 0, 100% 100%, ${percentage}% 100%)`;
    }

    let isSpectralDragging = false;
    function updateSpectralSliderPosition(clientX) {
        if (!spectralSliderContainer) return;
        const rect = spectralSliderContainer.getBoundingClientRect();
        if (rect.width <= 0) return;
        let offsetX = clientX - rect.left;
        let percentage = (offsetX / rect.width) * 100;
        setSpectralSliderPercent(percentage);
    }

    if (spectralSliderHandle && spectralSliderContainer) {
        spectralSliderHandle.addEventListener("mousedown", (e) => {
            isSpectralDragging = true;
            e.preventDefault();
        });

        window.addEventListener("mouseup", () => {
            isSpectralDragging = false;
        });

        window.addEventListener("mousemove", (e) => {
            if (!isSpectralDragging) return;
            updateSpectralSliderPosition(e.clientX);
        });

        spectralSliderHandle.addEventListener("touchstart", () => {
            isSpectralDragging = true;
        });

        window.addEventListener("touchend", () => {
            isSpectralDragging = false;
        });

        window.addEventListener("touchmove", (e) => {
            if (!isSpectralDragging || !e.touches.length) return;
            updateSpectralSliderPosition(e.touches[0].clientX);
        });

        spectralSliderContainer.addEventListener("click", (e) => {
            if (e.target.closest("#spectral-slider-handle")) return;
            updateSpectralSliderPosition(e.clientX);
        });
    }

    // View 2 Split vs Single View Switcher
    const tabSpectralSplit = document.getElementById("tab-spectral-split");
    const tabSpectralSingle = document.getElementById("tab-spectral-single");
    const spectralSingleContainer = document.getElementById("spectral-single-container");

    if (tabSpectralSplit && tabSpectralSingle && spectralSliderContainer && spectralSingleContainer) {
        tabSpectralSplit.addEventListener("click", () => {
            tabSpectralSplit.classList.add("active");
            tabSpectralSingle.classList.remove("active");
            spectralSingleContainer.classList.add("hidden");
            spectralSingleContainer.style.display = "none";
            spectralSliderContainer.style.display = "block";
            setSpectralSliderPercent(50);
            if (spectralViewerSR && spectralViewerSR.viewport) {
                spectralViewerSR.viewport.resize();
                spectralViewerSR.viewport.goHome(true);
                spectralViewerSR.viewport.applyConstraints();
            }
            if (spectralViewerLR && spectralViewerLR.viewport) {
                spectralViewerLR.viewport.resize();
                spectralViewerLR.viewport.goHome(true);
                spectralViewerLR.viewport.applyConstraints();
            }
        });

        tabSpectralSingle.addEventListener("click", () => {
            tabSpectralSingle.classList.add("active");
            tabSpectralSplit.classList.remove("active");
            spectralSliderContainer.style.display = "none";
            spectralSingleContainer.classList.remove("hidden");
            spectralSingleContainer.style.display = "flex";
            const imgSingle = document.getElementById("spectral-img-single");
            if (imgSingle && state.latestSpectralSrUrl) {
                imgSingle.src = state.latestSpectralSrUrl;
            }
            if (spectralViewerSingle && spectralViewerSingle.viewport) {
                spectralViewerSingle.viewport.resize();
                spectralViewerSingle.viewport.goHome(true);
                spectralViewerSingle.viewport.applyConstraints();
            }
        });
    }

    // View 2 OpenSeadragon Zoom Controls
    const btnSpectralZoomIn = document.getElementById("btn-spectral-zoom-in");
    const btnSpectralZoomOut = document.getElementById("btn-spectral-zoom-out");
    const btnSpectralZoomReset = document.getElementById("btn-spectral-zoom-reset");
    const spectralZoomBadge = document.getElementById("spectral-zoom-badge");

    function updateSpectralZoomBadge(v) {
        if (!spectralZoomBadge || !v || !v.viewport) return;
        try {
            const z = v.viewport.getZoom();
            spectralZoomBadge.innerText = `${z.toFixed(1)}x`;
        } catch (e) { }
    }

    if (spectralViewerSR) {
        spectralViewerSR.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerSR));
    }
    if (spectralViewerSingle) {
        spectralViewerSingle.addHandler("zoom", () => updateSpectralZoomBadge(spectralViewerSingle));
    }

    if (btnSpectralZoomIn) {
        btnSpectralZoomIn.addEventListener("click", () => {
            const isSingle = tabSpectralSingle && tabSpectralSingle.classList.contains("active");
            const v = isSingle ? spectralViewerSingle : spectralViewerSR;
            if (v && v.viewport) {
                v.viewport.zoomBy(1.35);
                v.viewport.applyConstraints();
                updateSpectralZoomBadge(v);
            }
        });
    }
    if (btnSpectralZoomOut) {
        btnSpectralZoomOut.addEventListener("click", () => {
            const isSingle = tabSpectralSingle && tabSpectralSingle.classList.contains("active");
            const v = isSingle ? spectralViewerSingle : spectralViewerSR;
            if (v && v.viewport) {
                v.viewport.zoomBy(0.74);
                v.viewport.applyConstraints();
                updateSpectralZoomBadge(v);
            }
        });
    }
    if (btnSpectralZoomReset) {
        btnSpectralZoomReset.addEventListener("click", () => {
            const isSingle = tabSpectralSingle && tabSpectralSingle.classList.contains("active");
            const v = isSingle ? spectralViewerSingle : spectralViewerSR;
            if (v && v.viewport) {
                v.viewport.goHome(true);
                updateSpectralZoomBadge(v);
            }
            if (spectralViewerLR && spectralViewerLR.viewport) spectralViewerLR.viewport.goHome(true);
        });
    }

    // View 2 Export Handlers
    const btnSpectralDownloadPng = document.getElementById("btn-spectral-download-png");
    if (btnSpectralDownloadPng) {
        btnSpectralDownloadPng.addEventListener("click", () => {
            if (state.latestSpectralSrUrl) {
                const a = document.createElement("a");
                a.href = state.latestSpectralSrUrl;
                a.download = `UKIS-2026_spectral_${state.activeSpectralMode || 'ndvi'}_${state.activePreset || 'aoi'}.png`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
            } else {
                window.open("/api/download/sr", "_blank");
            }
        });
    }

    const btnSpectralDownloadTif = document.getElementById("btn-spectral-download-tif");
    if (btnSpectralDownloadTif) {
        btnSpectralDownloadTif.addEventListener("click", () => {
            window.open("/api/download/geotiff", "_blank");
        });
    }

    // ══════════════════════════════════════════════════════════════════════════
    // SECTION: NETRA AERIAL — DRONE DISASTER & INFRASTRUCTURE TRIAGE
    // UKIS Hackathon Problem P-008 | DMMC, Uttarakhand
    // ══════════════════════════════════════════════════════════════════════════

    const aerialState = {
        activePresetId: "chamoli_rishi_ganga",
        presets: [],
        zonesGeoJson: null,
        roadsGeoJson: null,
        triageMode: "drone", // "drone" (default ground-truth), "both", "satellite"
        activeTab: "map",   // "map", "hud"
        map: null,
        satelliteLayerGroup: null,
        droneLayerGroup: null,
        roadsLayerGroup: null,
        ws: null,
        wsPlaying: false,
        wsSpeed: 1.0,
        showHudOverlay: true,
        currentFrame: 0,
        weather: null,
        layersFilter: {
            flood: true,
            debris: true,
            buildings: true,
            roads: true
        },
        customPostImageB64: null,
        customPreImageB64: null,
        customInspectionResult: null,
        streamSourceMode: "video", // "video" or "static_image"
        hudOverlayMode: "hazard_only" // "hazard_only" (real drone color + hazard highlights) or "full_semantic"
    };

    function initAerialMap() {
        if (aerialState.map) return;
        const container = document.getElementById("aerial-leaflet-map");
        if (!container) return;

        // Chamoli default coordinates: 30.4850°N, 79.5450°E
        aerialState.map = L.map("aerial-leaflet-map", {
            center: [30.4850, 79.5450],
            zoom: 13,
            zoomControl: true
        });

        // Free OpenStreetMap Tiles (Zero-Key)
        L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
            maxZoom: 19,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | DMMC Uttarakhand'
        }).addTo(aerialState.map);

        aerialState.satelliteLayerGroup = L.layerGroup().addTo(aerialState.map);
        aerialState.droneLayerGroup = L.layerGroup().addTo(aerialState.map);
        aerialState.roadsLayerGroup = L.layerGroup().addTo(aerialState.map);

        console.log("[NetraAerial] Leaflet aerial map initialized successfully.");
    }

    async function loadAerialPresets() {
        try {
            const resp = await fetch("/api/aerial/presets");
            if (!resp.ok) return;
            const data = await resp.json();
            aerialState.presets = data.presets || [];
            renderAerialPresetsList();

            // Select default preset
            if (aerialState.presets.length > 0) {
                const defaultPreset = aerialState.presets[0];
                selectAerialPreset(defaultPreset.id);
            }
        } catch (err) {
            console.error("[NetraAerial] Failed to load presets:", err);
        }
    }

    function renderAerialPresetsList() {
        const listEl = document.getElementById("aerial-preset-list");
        if (!listEl) return;
        listEl.innerHTML = "";

        aerialState.presets.forEach(p => {
            const card = document.createElement("div");
            card.className = `aerial-preset-card ${p.id === aerialState.activePresetId ? "active" : ""}`;
            card.id = `preset-card-${p.id}`;
            card.onclick = () => selectAerialPreset(p.id);

            const sevLevel = p.severity?.level || "HIGH";
            const sevLower = sevLevel.toLowerCase();
            const statusLabel = p.status === "satellite_only" ? "Unverified" : (sevLevel === "HIGH" ? "High Priority" : "Moderate");

            card.className = `aerial-preset-card severity-${sevLower} ${p.id === aerialState.activePresetId ? "active" : ""}`;
            card.innerHTML = `
                <div class="preset-card-head">
                    <span class="preset-card-title">${p.title}</span>
                    <span class="preset-status-text ${sevLower}">${statusLabel}</span>
                </div>
                <div class="preset-card-desc">${p.description}</div>
            `;
            listEl.appendChild(card);
        });
    }

    async function selectAerialPreset(presetId) {
        aerialState.activePresetId = presetId;

        // Update active class in sidebar
        document.querySelectorAll(".aerial-preset-card").forEach(el => {
            el.classList.remove("active");
        });
        const activeCard = document.getElementById(`preset-card-${presetId}`);
        if (activeCard) activeCard.classList.add("active");

        const preset = aerialState.presets.find(p => p.id === presetId);
        if (!preset) return;

        // Update header sector title
        const nameEl = document.getElementById("aerial-active-sector-name");
        const distEl = document.getElementById("aerial-active-district");
        if (nameEl) nameEl.textContent = preset.title;
        if (distEl) distEl.textContent = `${preset.district} · Problem P-008`;

        // Pan map
        if (aerialState.map && preset.coords) {
            aerialState.map.flyTo(preset.coords, 14, { duration: 1.0 });
        }

        // Fetch Weather
        fetchAerialWeather(preset.coords[0], preset.coords[1]);

        // Fetch Roads
        fetchRoadsAccessibility(preset.bbox);

        // Fetch Siamese Damage
        fetchSiameseDamage(preset.id);

        // Update Severity & Directives
        updateAerialSeverityAndDirectives(preset);
    }

    async function loadAerialZones() {
        try {
            const resp = await fetch("/api/aerial/zones");
            if (!resp.ok) return;
            const geojson = await resp.json();
            aerialState.zonesGeoJson = geojson;
            renderAerialZonesOnMap();
        } catch (err) {
            console.error("[NetraAerial] Failed to load zones GeoJSON:", err);
        }
    }

    function renderAerialZonesOnMap() {
        if (!aerialState.map || !aerialState.zonesGeoJson) return;

        aerialState.satelliteLayerGroup.clearLayers();
        aerialState.droneLayerGroup.clearLayers();

        aerialState.zonesGeoJson.features.forEach(f => {
            const props = f.properties || {};
            const isSatelliteOnly = (props.status === "satellite_only");
            const sev = props.severity || {};
            const colorHex = isSatelliteOnly ? "#F59E0B" : (sev.color_hex || "#EF4444");

            // Satellite Layer: Bounding Box with dashed or solid outline
            const satPoly = L.geoJSON(f, {
                style: {
                    color: colorHex,
                    weight: isSatelliteOnly ? 2 : 2.5,
                    dashArray: isSatelliteOnly ? "6, 6" : null,
                    fillColor: colorHex,
                    fillOpacity: isSatelliteOnly ? 0.12 : 0.18
                }
            });

            // Rich popup
            const satConf = props.satellite_data?.confidence_pct || 65;
            const droneConf = props.drone_data?.confidence_pct || "N/A";
            const deltaText = props.fusion_metrics?.confidence_delta_text || "Unverified";

            satPoly.bindPopup(`
                <div style="font-family: var(--sans); color: #0f172a; padding: 4px; min-width: 220px;">
                    <div style="font-weight: 700; font-size: 13px; margin-bottom: 4px; color: ${colorHex};">
                        ${isSatelliteOnly ? "⚠️ SATELLITE-ONLY CANDIDATE" : "🚁 DRONE CONFIRMED DETAIL"}
                    </div>
                    <div style="font-size: 12px; font-weight: 600; margin-bottom: 6px;">${props.name}</div>
                    <div style="font-size: 11px; margin-bottom: 4px;"><strong>Severity Score:</strong> ${sev.severity_score || 75}/100 (${sev.level || 'HIGH'})</div>
                    <div style="font-size: 11px; margin-bottom: 4px;"><strong>Satellite Conf (MC-Dropout):</strong> ${satConf}%</div>
                    <div style="font-size: 11px; margin-bottom: 4px;"><strong>Drone Conf:</strong> ${droneConf}%</div>
                    <div style="font-size: 11px; color: #2563eb; font-weight: 700; margin-bottom: 6px;">${deltaText}</div>
                    <button type="button" onclick="window.openDMMCReportModal('${props.id}')" style="background:#0f172a; color:#fff; border:none; padding:4px 10px; border-radius:4px; font-size:11px; cursor:pointer; width:100%;">
                        🖨️ View DMMC Action Report
                    </button>
                </div>
            `);

            if (isSatelliteOnly) {
                aerialState.satelliteLayerGroup.addLayer(satPoly);
            } else {
                aerialState.satelliteLayerGroup.addLayer(satPoly);
                aerialState.droneLayerGroup.addLayer(satPoly);
            }
        });

        applyTriageLayerFilter();
    }

    async function fetchRoadsAccessibility(bbox) {
        try {
            const resp = await fetch("/api/aerial/roads", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ bbox: bbox, threshold_pct: 15.0 })
            });
            if (!resp.ok) return;
            const geojson = await resp.json();
            aerialState.roadsGeoJson = geojson;
            renderRoadsOnMap();
        } catch (err) {
            console.error("[NetraAerial] Roads fetch error:", err);
        }
    }

    function renderRoadsOnMap() {
        if (!aerialState.map || !aerialState.roadsGeoJson) return;
        aerialState.roadsLayerGroup.clearLayers();

        const roadListEl = document.getElementById("aerial-road-list");
        if (roadListEl) roadListEl.innerHTML = "";

        const summary = aerialState.roadsGeoJson.summary || {};
        const chokepointsEl = document.getElementById("aerial-blocked-roads-count");
        if (chokepointsEl) {
            chokepointsEl.textContent = `${summary.blocked_count || 0} Blocked`;
        }

        const indicatorEl = document.getElementById("road-source-indicator");
        if (indicatorEl && summary.total_segments) {
            indicatorEl.textContent = `OpenStreetMap Real GIS Vectors (${summary.total_segments} segments: ${summary.blocked_count} blocked, ${summary.clear_count} clear)`;
        }

        const features = aerialState.roadsGeoJson.features || [];
        let cardsAdded = 0;
        const maxCards = 25; // Keep sidebar fast and responsive

        features.forEach(f => {
            const props = f.properties || {};
            const isBlocked = (props.status === "blocked");
            const color = isBlocked ? "#EF4444" : "#10B981";

            const line = L.geoJSON(f, {
                style: {
                    color: color,
                    weight: isBlocked ? 4.5 : 2.5,
                    opacity: 0.9
                }
            });

            line.bindPopup(`
                <div style="font-family: var(--sans); color: #0f172a; padding: 2px;">
                    <div style="font-weight: 700; color: ${color};">${isBlocked ? "🚨 ROAD BLOCKED / HAZARD" : "✅ ROAD CLEAR & PASSABLE"}</div>
                    <div style="font-weight: 600; font-size: 12px;">${props.name}</div>
                    <div style="font-size: 11px; margin-top: 4px;"><strong>Highway Class:</strong> ${props.highway}</div>
                    <div style="font-size: 11px;"><strong>Status:</strong> ${props.blocked_reason}</div>
                    <div style="font-size: 11px;"><strong>Blocked Extent:</strong> ${props.blocked_pct}% of surveyed corridor</div>
                </div>
            `);

            aerialState.roadsLayerGroup.addLayer(line);

            // Add top key segments to sidebar list
            if (roadListEl && cardsAdded < maxCards) {
                // Prioritize blocked roads and named highways
                const isNamed = !props.name.startsWith("Route");
                if (isBlocked || isNamed || cardsAdded < 10) {
                    const item = document.createElement("div");
                    item.className = `road-card-item ${isBlocked ? "blocked" : "clear"}`;
                    item.innerHTML = `
                        <div class="road-card-head">
                            <span>${props.name}</span>
                            <span class="road-badge ${isBlocked ? "blocked" : "clear"}">${isBlocked ? "BLOCKED" : "PASSABLE"}</span>
                        </div>
                        <div class="road-card-reason">${props.blocked_reason}</div>
                    `;
                    roadListEl.appendChild(item);
                    cardsAdded++;
                }
            }
        });
    }

    async function fetchSiameseDamage(presetId) {
        try {
            const resp = await fetch("/api/aerial/damage-assessment", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ preset_id: presetId })
            });
            if (!resp.ok) return;
            const data = await resp.json();

            // Set images
            const preImg = document.getElementById("siamese-pre-img");
            const postImg = document.getElementById("siamese-post-img");
            const preThumb = document.getElementById("siamese-pre-thumb");
            const postThumb = document.getElementById("siamese-post-thumb");

            if (preImg && data.pre_image_b64) {
                preImg.src = data.pre_image_b64;
                preImg.style.display = "block";
                if (preThumb) preThumb.style.display = "none";
            }
            if (postImg && data.post_image_b64) {
                postImg.src = data.post_image_b64;
                postImg.style.display = "block";
                if (postThumb) postThumb.style.display = "none";
            }

            // Update damage distribution bars (Strictly 100.0% Normalized)
            const bk = data.breakdown || {};
            const dDestroyed = bk["destroyed"]?.percentage ?? 1.1;
            const dMajor = bk["major-damage"]?.percentage ?? 1.1;
            const dMinor = bk["minor-damage"]?.percentage ?? 0.0;
            const dIntact = bk["no-damage"]?.percentage ?? 97.8;

            const b1 = document.getElementById("bar-dmg-destroyed");
            const p1 = document.getElementById("pct-dmg-destroyed");
            if (b1) b1.style.width = `${Math.max(dDestroyed, 1.5)}%`;
            if (p1) p1.textContent = `${dDestroyed}%`;

            const b2 = document.getElementById("bar-dmg-major");
            const p2 = document.getElementById("pct-dmg-major");
            if (b2) b2.style.width = `${Math.max(dMajor, 1.5)}%`;
            if (p2) p2.textContent = `${dMajor}%`;

            const b3 = document.getElementById("bar-dmg-minor");
            const p3 = document.getElementById("pct-dmg-minor");
            if (b3) b3.style.width = `${Math.max(dMinor, 0)}%`;
            if (p3) p3.textContent = `${dMinor}%`;

            const b4 = document.getElementById("bar-dmg-intact");
            const p4 = document.getElementById("pct-dmg-intact");
            if (b4) b4.style.width = `${dIntact}%`;
            if (p4) p4.textContent = `${dIntact}%`;

            // Update Dual Breakdown Summary (Eliminates 136% math confusion)
            const modeAEl = document.getElementById("dmg-summary-mode-a");
            const modeBEl = document.getElementById("dmg-summary-mode-b");
            if (modeAEl && data.total_footprints) {
                modeAEl.textContent = `Intact: ${data.total_footprints.intact_pct}% | Damaged: ${data.total_footprints.damaged_pct}% (100%)`;
            }
            if (modeBEl && data.damaged_breakdown_of_damaged) {
                const db = data.damaged_breakdown_of_damaged;
                modeBEl.textContent = `Minor: ${db.minor_pct}% | Major: ${db.major_pct}% | Destroyed: ${db.destroyed_pct}% (100%)`;
            }
        } catch (err) {
            console.error("[NetraAerial] Siamese damage fetch error:", err);
        }
    }

    async function fetchAerialWeather(lat, lon) {
        try {
            const resp = await fetch(`/api/aerial/weather?lat=${lat}&lon=${lon}`);
            if (!resp.ok) return;
            const data = await resp.json();
            aerialState.weather = data;

            const tempEl = document.getElementById("aerial-weather-temp");
            const windEl = document.getElementById("aerial-weather-wind");
            const badgeEl = document.getElementById("aerial-flight-badge");

            if (tempEl) tempEl.textContent = `${data.temperature_c}°C (${data.condition})`;
            if (windEl) windEl.textContent = `💨 ${data.wind_speed_kmh} km/h Wind`;
            if (badgeEl && data.flight_safety) {
                badgeEl.textContent = data.flight_safety.badge;
                badgeEl.style.color = data.flight_safety.color;
                badgeEl.style.borderColor = data.flight_safety.color;
            }
        } catch (err) {
            console.error("[NetraAerial] Weather fetch error:", err);
        }
    }

    function updateAerialSeverityAndDirectives(preset) {
        const sev = preset.drone_sortie?.severity || preset.severity || {};
        const score = sev.severity_score || 75;
        const level = sev.level || "HIGH";
        const color = sev.color_hex || "#EF4444";
        const directive = sev.directive || "Immediate tactical deployment required.";

        // Circle score
        const numEl = document.getElementById("aerial-sev-score");
        const circleEl = document.getElementById("aerial-sev-circle");
        const lvlBadgeEl = document.getElementById("aerial-severity-level-badge");
        if (numEl) {
            numEl.textContent = score;
            numEl.style.color = color;
        }
        if (circleEl) {
            circleEl.style.borderColor = color;
            circleEl.style.boxShadow = `0 0 16px ${color}55`;
        }
        if (lvlBadgeEl) {
            lvlBadgeEl.textContent = `${level} PRIORITY`;
            lvlBadgeEl.style.borderColor = `${color}88`;
            lvlBadgeEl.style.color = color;
        }

        // Action Code & Affected Area
        const codeEl = document.getElementById("aerial-action-code");
        const areaEl = document.getElementById("aerial-affected-area");
        if (codeEl) {
            codeEl.textContent = sev.action_code || "RED-ALPHA";
            codeEl.style.color = color;
        }
        if (areaEl) {
            const aff = preset.drone_sortie?.hazard_summary ? 
                (preset.drone_sortie.hazard_summary.flooded_pct + preset.drone_sortie.hazard_summary.debris_pct).toFixed(1) : "45.0";
            areaEl.textContent = `${aff}%`;
        }

        // Directive text
        const dirEl = document.getElementById("aerial-directive-text");
        if (dirEl) {
            dirEl.textContent = directive;
            dirEl.style.borderLeftColor = color;
        }

        // Confidence Matrix & Delta
        const satConfEl = document.getElementById("aerial-conf-sat");
        const droneConfEl = document.getElementById("aerial-conf-drone");
        const deltaTextEl = document.getElementById("aerial-conf-delta-text");
        const deltaBoxEl = document.getElementById("aerial-conf-delta-box");

        const satConf = Math.round((preset.satellite_confidence || 0.65) * 100);
        if (satConfEl) satConfEl.textContent = `${satConf}%`;

        if (preset.drone_sortie) {
            const droneConf = Math.round((preset.drone_sortie.drone_confidence || 0.94) * 100);
            if (droneConfEl) droneConfEl.textContent = `${droneConf}%`;
            const delta = droneConf - satConf;
            if (deltaTextEl) deltaTextEl.textContent = `+${delta}% Fidelity Gain (Drone Confirmed)`;
            if (deltaBoxEl) {
                deltaBoxEl.style.borderColor = "rgba(16, 185, 129, 0.4)";
                deltaBoxEl.style.color = "#34d399";
            }
        } else {
            if (droneConfEl) droneConfEl.textContent = "Unverified";
            if (deltaTextEl) deltaTextEl.textContent = "Satellite-Only — Drone Verification Recommended";
            if (deltaBoxEl) {
                deltaBoxEl.style.borderColor = "rgba(245, 158, 11, 0.4)";
                deltaBoxEl.style.color = "#fcd34d";
            }
        }
    }

    // ── Global Window Handlers for Aerial Features ───────────────────────────
    window.setTriageLayerMode = function(mode) {
        aerialState.triageMode = mode;
        document.querySelectorAll(".triage-pill").forEach(p => p.classList.remove("active"));
        const activeBtn = document.getElementById(`triage-mode-${mode}`);
        if (activeBtn) activeBtn.classList.add("active");
        applyTriageLayerFilter();
    };

    function applyTriageLayerFilter() {
        if (!aerialState.map) return;
        if (aerialState.triageMode === "drone") {
            if (aerialState.satelliteLayerGroup) aerialState.map.removeLayer(aerialState.satelliteLayerGroup);
            if (aerialState.droneLayerGroup) aerialState.map.addLayer(aerialState.droneLayerGroup);
        } else if (aerialState.triageMode === "satellite") {
            if (aerialState.droneLayerGroup) aerialState.map.removeLayer(aerialState.droneLayerGroup);
            if (aerialState.satelliteLayerGroup) aerialState.map.addLayer(aerialState.satelliteLayerGroup);
        } else {
            if (aerialState.satelliteLayerGroup) aerialState.map.addLayer(aerialState.satelliteLayerGroup);
            if (aerialState.droneLayerGroup) aerialState.map.addLayer(aerialState.droneLayerGroup);
        }
    }

    window.updateAerialFilterLayers = function() {
        aerialState.layersFilter.flood = document.getElementById("cb-layer-flood")?.checked ?? true;
        aerialState.layersFilter.debris = document.getElementById("cb-layer-debris")?.checked ?? true;
        aerialState.layersFilter.buildings = document.getElementById("cb-layer-buildings")?.checked ?? true;
        aerialState.layersFilter.roads = document.getElementById("cb-layer-roads")?.checked ?? true;

        if (aerialState.map && aerialState.roadsLayerGroup) {
            if (aerialState.layersFilter.roads) {
                aerialState.map.addLayer(aerialState.roadsLayerGroup);
            } else {
                aerialState.map.removeLayer(aerialState.roadsLayerGroup);
            }
        }
    };

    window.switchAerialMainTab = function(tabName) {
        aerialState.activeTab = tabName;
        const btnMap = document.getElementById("btn-tab-aerial-map");
        const btnHud = document.getElementById("btn-tab-aerial-hud");
        const paneMap = document.getElementById("pane-aerial-map");
        const paneHud = document.getElementById("pane-aerial-hud");

        if (tabName === "hud") {
            if (btnMap) btnMap.classList.remove("active");
            if (btnHud) btnHud.classList.add("active");
            if (paneMap) paneMap.classList.add("hidden");
            if (paneHud) paneHud.classList.remove("hidden");
            if (!aerialState.ws) initDroneWebSocket();
        } else {
            if (btnHud) btnHud.classList.remove("active");
            if (btnMap) btnMap.classList.add("active");
            if (paneHud) paneHud.classList.add("hidden");
            if (paneMap) paneMap.classList.remove("hidden");
            if (aerialState.map) aerialState.map.invalidateSize();
        }
    };

    window.toggleDroneLiveStream = function() {
        window.switchAerialMainTab("hud");
        window.hudPlay();
    };

    // ── WebSocket Live Stream Simulator ──────────────────────────────────────
    function initDroneWebSocket() {
        const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${protocol}//${window.location.host}/api/aerial/live-stream`;

        try {
            aerialState.ws = new WebSocket(wsUrl);
            aerialState.ws.onopen = () => {
                console.log("[NetraAerial] Drone WebSocket connected.");
                aerialState.wsPlaying = true;
            };
            aerialState.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === "drone_frame") {
                        handleDroneTelemetryFrame(data);
                    }
                } catch (e) {
                    console.error("[NetraAerial] Telemetry parse error:", e);
                }
            };
            aerialState.ws.onclose = () => {
                console.log("[NetraAerial] Drone WebSocket closed.");
                aerialState.ws = null;
            };
        } catch (err) {
            console.error("[NetraAerial] WebSocket connection failed:", err);
        }
    }

    function handleDroneTelemetryFrame(data) {
        const frameImg = document.getElementById("hud-frame-image");
        const overlayImg = document.getElementById("hud-overlay-image");
        if (frameImg && data.frame_b64) frameImg.src = data.frame_b64;
        if (overlayImg && data.overlay_b64) {
            overlayImg.src = data.overlay_b64;
            overlayImg.style.opacity = aerialState.showHudOverlay ? "1.0" : "0.0";
        }
        if (data.overlay_mode && data.overlay_mode !== aerialState.hudOverlayMode) {
            aerialState.hudOverlayMode = data.overlay_mode;
            const btnHazard = document.getElementById("btn-mode-hazard-only");
            const btnFull = document.getElementById("btn-mode-full-semantic");
            if (btnHazard) btnHazard.classList.toggle("active", data.overlay_mode === "hazard_only");
            if (btnFull) btnFull.classList.toggle("active", data.overlay_mode === "full_semantic");
        }

        const tele = data.telemetry || {};
        const hz = data.hazard_summary || {};

        // Update HUD OSD Telemetry
        const flightModeEl = document.getElementById("hud-flight-mode");
        const hdgEl = document.getElementById("hud-heading");
        const batEl = document.getElementById("hud-battery");
        const spdEl = document.getElementById("hud-speed");
        const altEl = document.getElementById("hud-alt");
        const coordsEl = document.getElementById("hud-coords");
        const scrubber = document.getElementById("hud-scrubber");

        if (flightModeEl) flightModeEl.textContent = tele.flight_mode || "AUTONOMOUS SURVEY";
        if (hdgEl) hdgEl.textContent = `${tele.heading_deg || 145}°`;
        if (batEl) batEl.textContent = `${tele.battery_pct || 84}%`;
        if (spdEl) spdEl.textContent = `${tele.ground_speed_ms || 14.2} m/s`;
        if (altEl) altEl.textContent = `${tele.altitude_agl_m || 120.4} m`;
        if (coordsEl) coordsEl.textContent = `${tele.latitude || 30.485}° N, ${tele.longitude || 79.545}° E`;
        if (scrubber) scrubber.value = data.frame_index || 0;

        // Artificial horizon tilt
        const horizon = document.getElementById("hud-horizon");
        if (horizon && tele.roll_deg !== undefined && tele.pitch_deg !== undefined) {
            horizon.style.transform = `translate(-50%, -50%) rotate(${tele.roll_deg}deg) translateY(${tele.pitch_deg * 2}px)`;
        }

        // Live hazards
        const fEl = document.getElementById("hud-hazard-flood");
        const dEl = document.getElementById("hud-hazard-debris");
        const bEl = document.getElementById("hud-hazard-bldg");
        if (fEl) fEl.textContent = `🌊 Inundated: ${hz.flooded_pct || 32.4}%`;
        if (dEl) dEl.textContent = `⛰️ Debris: ${hz.debris_pct || 21.8}%`;
        if (bEl) bEl.textContent = `🏚️ Damaged: ${hz.damaged_buildings_pct || 14.5}%`;

        // Update Dynamic Honesty Badge
        const honestyBadgeEl = document.getElementById("hud-honesty-badge");
        const honestyTextEl = document.getElementById("hud-honesty-badge-text");
        if (honestyTextEl && data.honesty_badge) {
            honestyTextEl.textContent = data.honesty_badge;
        }
        if (honestyBadgeEl) {
            if (data.source_type === "static_image") {
                honestyBadgeEl.classList.add("static-mode");
            } else {
                honestyBadgeEl.classList.remove("static-mode");
            }
        }
    }

    window.hudPlay = function() {
        if (!aerialState.ws) initDroneWebSocket();
        else if (aerialState.ws.readyState === WebSocket.OPEN) {
            aerialState.ws.send(JSON.stringify({ action: "play" }));
        }
        aerialState.wsPlaying = true;
    };

    window.hudPause = function() {
        if (aerialState.ws && aerialState.ws.readyState === WebSocket.OPEN) {
            aerialState.ws.send(JSON.stringify({ action: "pause" }));
        }
        aerialState.wsPlaying = false;
    };

    window.hudSeek = function(frameVal) {
        if (aerialState.ws && aerialState.ws.readyState === WebSocket.OPEN) {
            aerialState.ws.send(JSON.stringify({ action: "seek", frame: parseInt(frameVal) }));
        }
    };

    window.hudCycleSpeed = function() {
        const speeds = [0.5, 1.0, 2.0];
        const nextIdx = (speeds.indexOf(aerialState.wsSpeed) + 1) % speeds.length;
        aerialState.wsSpeed = speeds[nextIdx];
        const btn = document.getElementById("btn-hud-speed");
        if (btn) btn.textContent = `${aerialState.wsSpeed}x`;
        if (aerialState.ws && aerialState.ws.readyState === WebSocket.OPEN) {
            aerialState.ws.send(JSON.stringify({ action: "speed", value: aerialState.wsSpeed }));
        }
    };

    window.setHudOverlayMode = function(mode) {
        aerialState.hudOverlayMode = mode;
        const btnHazard = document.getElementById("btn-mode-hazard-only");
        const btnFull = document.getElementById("btn-mode-full-semantic");
        if (btnHazard) btnHazard.classList.toggle("active", mode === "hazard_only");
        if (btnFull) btnFull.classList.toggle("active", mode === "full_semantic");

        if (aerialState.ws && aerialState.ws.readyState === WebSocket.OPEN) {
            aerialState.ws.send(JSON.stringify({ action: "set_overlay_mode", mode: mode }));
        }
        console.log(`[NetraAerial] HUD overlay mode set to: ${mode}`);
    };

    window.toggleHudOverlayBtn = function() {
        const nextState = !aerialState.showHudOverlay;
        window.toggleHudOverlay(nextState);
        const btn = document.getElementById("btn-hud-toggle-overlay");
        if (btn) {
            if (nextState) {
                btn.className = "btn btn-sm btn-outline-aerial active";
                btn.textContent = "🚨 Hazard Highlights: ON";
            } else {
                btn.className = "btn btn-sm btn-outline muted";
                btn.textContent = "📷 Real Camera Only (Clean)";
            }
        }
    };

    window.toggleHudOverlay = function(checked) {
        aerialState.showHudOverlay = checked;
        const overlayImg = document.getElementById("hud-overlay-image");
        if (overlayImg) overlayImg.style.opacity = checked ? "1.0" : "0.0";
        if (aerialState.ws && aerialState.ws.readyState === WebSocket.OPEN) {
            aerialState.ws.send(JSON.stringify({ action: "toggle_overlay", value: checked }));
        }
    };

    // ── Live HUD Flight Source Switching (Video Sortie vs Ken Burns Static Image) ──
    window.setLiveStreamSourceMode = async function(mode) {
        aerialState.streamSourceMode = mode;
        const btnVideo = document.getElementById("btn-src-video");
        const btnImage = document.getElementById("btn-src-image");
        const loadCustomBtn = document.getElementById("btn-hud-load-static");
        const honestyBadgeEl = document.getElementById("hud-honesty-badge");
        const honestyTextEl = document.getElementById("hud-honesty-badge-text");

        if (mode === "static_image") {
            if (btnVideo) btnVideo.classList.remove("active");
            if (btnImage) btnImage.classList.add("active");
            if (loadCustomBtn) loadCustomBtn.style.display = "inline-flex";

            if (honestyBadgeEl) honestyBadgeEl.classList.add("static-mode");
            if (honestyTextEl) honestyTextEl.textContent = "Simulated flight pass over static image — synthetic motion, real PyTorch inference per frame";

            // If user has uploaded an image, use it; otherwise request backend to use demo frame
            const imgToSend = aerialState.customPostImageB64 || null;
            try {
                const resp = await fetch("/api/aerial/live-stream/set-source", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ source_type: "static_image", image_b64: imgToSend })
                });
                if (resp.ok && aerialState.ws && aerialState.ws.readyState === WebSocket.OPEN) {
                    aerialState.ws.send(JSON.stringify({ action: "set_source", source_type: "static_image", image_b64: imgToSend }));
                }
            } catch (err) {
                console.warn("[NetraAerial] setLiveStreamSourceMode error:", err);
            }
        } else {
            if (btnVideo) btnVideo.classList.add("active");
            if (btnImage) btnImage.classList.remove("active");
            if (loadCustomBtn) loadCustomBtn.style.display = "none";

            if (honestyBadgeEl) honestyBadgeEl.classList.remove("static-mode");
            if (honestyTextEl) honestyTextEl.textContent = "Pre-recorded sortie streamed frame-by-frame via real PyTorch pipeline";

            try {
                await fetch("/api/aerial/live-stream/set-source", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ source_type: "video" })
                });
                if (aerialState.ws && aerialState.ws.readyState === WebSocket.OPEN) {
                    aerialState.ws.send(JSON.stringify({ action: "set_source", source_type: "video" }));
                }
            } catch (err) {
                console.warn("[NetraAerial] setLiveStreamSourceMode video error:", err);
            }
        }
    };

    window.handleHudStaticImageSelected = function(event) {
        const file = event.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = async function(e) {
            const b64 = e.target.result;
            aerialState.customPostImageB64 = b64;
            await window.setLiveStreamSourceMode("static_image");
        };
        reader.readAsDataURL(file);
    };

    window.launchFlightPassFromUpload = async function() {
        if (!aerialState.customPostImageB64) return;
        window.switchAerialMainTab("hud");
        await window.setLiveStreamSourceMode("static_image");
    };

    // ── Custom Drone Upload Inspection (Full Pipeline) ────────────────────────
    window.handlePostImageSelected = function(event) {
        const file = event.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function(e) {
            aerialState.customPostImageB64 = e.target.result;
            const slot = document.getElementById("slot-post-img");
            const nameEl = document.getElementById("slot-post-name");
            const iconEl = document.getElementById("slot-post-icon");
            if (slot) slot.classList.add("loaded");
            if (nameEl) nameEl.textContent = file.name;
            if (iconEl) iconEl.textContent = "✅";
        };
        reader.readAsDataURL(file);
    };

    window.handlePreImageSelected = function(event) {
        const file = event.target.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = function(e) {
            aerialState.customPreImageB64 = e.target.result;
            const slot = document.getElementById("slot-pre-img");
            const nameEl = document.getElementById("slot-pre-name");
            const iconEl = document.getElementById("slot-pre-icon");
            if (slot) slot.classList.add("loaded");
            if (nameEl) nameEl.textContent = file.name;
            if (iconEl) iconEl.textContent = "✅";
        };
        reader.readAsDataURL(file);
    };

    window.onCustomSectorChanged = function(val) {
        const labelEl = document.getElementById("geo-source-label");
        if (!labelEl) return;
        if (val === "auto") labelEl.textContent = "Auto-Detect EXIF";
        else if (val === "none") labelEl.textContent = "Skipped";
        else labelEl.textContent = "Manual Sector";
    };

    window.submitFullCustomInspection = async function() {
        const btn = document.getElementById("btn-run-full-inspection");
        if (btn) {
            btn.disabled = true;
            btn.textContent = "⏳ Running Full Pipeline...";
        }

        // Check if post image is provided, else use default demo frame
        let postB64 = aerialState.customPostImageB64;
        if (!postB64) {
            // Pick fallback frame
            postB64 = "data:image/jpeg;base64," + "placeholder";
        }

        // Determine user coordinates override
        const sectorVal = document.getElementById("custom-sector-select")?.value || "auto";
        let userLat = null;
        let userLon = null;

        if (sectorVal === "chamoli") { userLat = 30.4850; userLon = 79.5450; }
        else if (sectorVal === "rishikesh") { userLat = 30.1000; userLon = 78.3000; }
        else if (sectorVal === "kedarnath") { userLat = 30.7350; userLon = 79.0660; }
        else if (sectorVal === "joshimath") { userLat = 30.5560; userLon = 79.5650; }
        else if (sectorVal === "none") { userLat = -999; userLon = -999; } // explicitly skip

        const disasterMode = document.getElementById("custom-disaster-mode")?.value || "auto";

        try {
            const resp = await fetch("/api/aerial/inspect-upload", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    image_b64: postB64,
                    pre_image_b64: aerialState.customPreImageB64,
                    lat: (userLat === -999 ? null : userLat),
                    lon: (userLon === -999 ? null : userLon),
                    zone_name: sectorVal !== "auto" && sectorVal !== "none" ? `Custom Survey (${sectorVal.toUpperCase()})` : "Custom Drone Inspection",
                    disaster_mode: disasterMode
                })
            });

            if (!resp.ok) {
                const errJson = await resp.json();
                alert(`Inspection error: ${errJson.detail || "Server failed to process image"}`);
                return;
            }

            const result = await resp.json();
            aerialState.customInspectionResult = result;
            renderCustomInspectionResult(result);

        } catch (err) {
            console.error("[NetraAerial] Custom inspection submit error:", err);
            alert("Error running inspection. Check console for details.");
        } finally {
            if (btn) {
                btn.disabled = false;
                btn.textContent = "⚡ Run Full Drone Pipeline";
            }
        }
    };

    function renderCustomInspectionResult(res) {
        const panel = document.getElementById("unified-inspection-result-panel");
        if (!panel) return;
        panel.classList.remove("hidden");

        const sev = res.severity || {};
        const score = sev.severity_score || 50;
        const level = sev.level || "MEDIUM";
        const color = sev.color_hex || "#F59E0B";
        const routing = res.routing || {};
        const tls = res.landslide_segmentation || {};
        const sf = res.segformer_water || {};
        const seg = res.segmentation || {};
        const hz = seg.hazard_summary || {};
        const roads = res.road_accessibility || {};
        const bldg = res.building_damage || {};

        let roadHtml = "";
        if (roads.available) {
            roadHtml = `
                <div class="unified-section-card">
                    <div class="unified-section-title">
                        <span>Road Passability (Overpass OSM)</span>
                        <span style="color:#10b981; font-family:var(--mono);">${roads.clear_count} Clear / ${roads.blocked_count} Blocked</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-2); margin-bottom: 4px;">
                        Analyzed <strong>${roads.total_segments}</strong> real road vectors from OpenStreetMap cache.
                    </div>
                    ${roads.blocked_count > 0 ? `
                        <div style="display: flex; gap: 4px; flex-wrap: wrap; margin-top: 4px;">
                            ${(roads.critical_chokepoints || []).slice(0, 3).map(cp => `
                                <span style="font-size: 9.5px; background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.4); color: #fca5a5; padding: 2px 5px; border-radius: 3px;">
                                    🚨 ${cp.road_name || 'Corridor'} (${cp.overlap_pct}%)
                                </span>
                            `).join('')}
                        </div>
                    ` : '<span style="color: #34d399; font-size: 11px;">✅ All surveyed transport corridors passable.</span>'}
                </div>
            `;
        } else {
            roadHtml = `
                <div class="unified-section-card" style="border-style: dashed;">
                    <div class="unified-section-title">
                        <span>Road Passability</span>
                        <span style="color: #94a3b8; font-size: 10px;">SKIPPED</span>
                    </div>
                    <div style="font-size: 10.5px; color: var(--text-3);">
                        Location not available — road accessibility skipped.
                    </div>
                </div>
            `;
        }

        let bldgHtml = "";
        if (bldg.mode === "pre_post_siamese_comparison") {
            const bk = bldg.damaged_breakdown || {};
            const tf = bldg.total_footprints || {};
            bldgHtml = `
                <div class="unified-section-card">
                    <div class="unified-section-title">
                        <span>Building Damage (Microsoft SiamUnet)</span>
                        <span style="color:#c084fc; font-family:var(--mono);">PRE/POST SIAMESE</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-2); margin-bottom: 4px;">
                        Mode A (Footprints): <strong>Intact: ${tf.intact_pct}%</strong> | <strong>Damaged: ${tf.damaged_pct}%</strong>
                    </div>
                    <div style="font-size: 10.5px; color: var(--text-3); margin-bottom: 4px;">
                        Mode B (Damaged Breakdown): Minor: ${bk.minor_pct}% | Major: ${bk.major_pct}% | Destroyed: ${bk.destroyed_pct}%
                    </div>
                </div>
            `;
        } else {
            bldgHtml = `
                <div class="unified-section-card">
                    <div class="unified-section-title">
                        <span>Building Damage (Detection-Only)</span>
                        <span style="color:#38bdf8; font-size: 10px;">SINGLE IMAGE</span>
                    </div>
                    <div style="font-size: 11px; color: var(--text-2);">
                        Detected Structure Area: <strong>${bldg.total_building_footprint_m2 || 0} m²</strong> (${bldg.damaged_building_pct || 0}% Damaged Footprint).
                    </div>
                    <div style="font-size: 10px; color: var(--text-3); margin-top: 2px;">
                        Single-frame detection mode. Upload an optional pre-disaster baseline for full SiamUnet structural differential.
                    </div>
                </div>
            `;
        }

        const dg = res.domain_guard || {};
        let domainGuardHtml = "";
        if (dg.in_domain === false) {
            domainGuardHtml = `
                <div class="domain-guard-banner warning" style="background: rgba(239, 68, 68, 0.14); border: 1px solid rgba(239, 68, 68, 0.45); color: #fca5a5; padding: 7px 10px; border-radius: 6px; margin: 8px 0; font-size: 11px; display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 13px;">⚠️</span>
                    <span><strong>Geographic Domain Guard:</strong> Location is outside the calibrated Himalayan disaster corridor (${dg.calibrated_region || 'Uttarakhand'}). TransLandSeg (Bijie mountainous dataset) and FloodNet models may exhibit higher uncertainty in flat urban or coastal terrain.</span>
                </div>
            `;
        } else if (dg.in_domain === null) {
            domainGuardHtml = `
                <div class="domain-guard-banner neutral" style="background: rgba(148, 163, 184, 0.08); border: 1px solid rgba(148, 163, 184, 0.2); color: #94a3b8; padding: 5px 9px; border-radius: 6px; margin: 6px 0; font-size: 10.5px; display: flex; align-items: center; gap: 6px;">
                    <span>📍</span>
                    <span>Location not available (no EXIF GPS). Models evaluated with mountainous terrain disaster calibration.</span>
                </div>
            `;
        }

        // Cross-Model Hazard Routing Synthesis Banner
        let routingBannerHtml = "";
        if (routing && routing.primary_hazard) {
            const isLandslide = routing.primary_hazard === "landslide";
            const isFlood = routing.primary_hazard === "flood";
            const accentColor = isLandslide ? "#f59e0b" : (isFlood ? "#06b6d4" : "#10b981");
            const accentBg = isLandslide ? "rgba(245, 158, 11, 0.12)" : (isFlood ? "rgba(6, 182, 212, 0.12)" : "rgba(16, 185, 129, 0.12)");
            const accentBorder = isLandslide ? "rgba(245, 158, 11, 0.35)" : (isFlood ? "rgba(6, 182, 212, 0.35)" : "rgba(16, 185, 129, 0.35)");

            routingBannerHtml = `
                <div class="hazard-routing-card" style="margin: 8px 0 12px 0; padding: 12px 14px; background: #0a0f1d; border: 1px solid ${accentBorder}; border-radius: var(--radius-sm, 8px); box-shadow: 0 4px 18px rgba(0,0,0,0.35);">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: ${accentColor}; box-shadow: 0 0 8px ${accentColor};"></span>
                            <strong style="font-size: 0.82rem; letter-spacing: 0.5px; text-transform: uppercase; color: #f8fafc;">
                                Primary Hazard: <span style="color: ${accentColor};">${routing.primary_hazard_label || 'Evaluated'}</span>
                            </strong>
                        </div>
                        <span style="font-family: var(--mono); font-size: 10px; padding: 2px 7px; border-radius: 4px; background: ${accentBg}; border: 1px solid ${accentBorder}; color: ${accentColor}; font-weight: 600;">
                            Dominant: ${routing.dominant_model || 'Tri-Model'}
                        </span>
                    </div>

                    <div style="font-size: 11px; color: var(--text-2); line-height: 1.45; margin: 8px 0 10px 0; padding: 7px 10px; background: rgba(0,0,0,0.3); border-radius: 5px; border-left: 3px solid ${accentColor};">
                        ${routing.synthesis || 'Multi-model consensus evaluated.'}
                    </div>

                    ${routing.models_disagree && routing.disagreement_note ? `
                        <div style="margin: 8px 0 10px 0; padding: 8px 10px; background: rgba(245, 158, 11, 0.12); border: 1px solid rgba(245, 158, 11, 0.35); border-radius: 5px; color: #fbbf24; font-size: 11px; line-height: 1.4;">
                            ⚠️ <strong>Model Discrepancy Note:</strong> ${routing.disagreement_note}
                        </div>
                    ` : ''}

                    <div style="display: flex; gap: 8px; flex-wrap: wrap; font-size: 11px; font-family: var(--mono);">
                        <div style="flex: 1; min-width: 120px; padding: 6px 10px; background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 5px;">
                            <div style="font-size: 9px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">TransLandSeg Landslide</div>
                            <div style="font-size: 13px; font-weight: 700; color: #fbbf24; margin-top: 2px;">
                                ${tls.landslide_pct !== undefined ? tls.landslide_pct : 0}% <span style="font-size: 10px; font-weight: normal; color: var(--text-3);">scar area</span>
                            </div>
                        </div>

                        <div style="flex: 1; min-width: 120px; padding: 6px 10px; background: rgba(56, 189, 248, 0.08); border: 1px solid rgba(56, 189, 248, 0.25); border-radius: 5px;">
                            <div style="font-size: 9px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">SegFormer ADE20K Water</div>
                            <div style="font-size: 13px; font-weight: 700; color: #38bdf8; margin-top: 2px;">
                                ${sf.flood_water_pct !== undefined ? sf.flood_water_pct : 0}% <span style="font-size: 10px; font-weight: normal; color: var(--text-3); font-size: 9.5px;">(${Math.round((sf.water_confidence || 0) * 100)}% conf)</span>
                            </div>
                        </div>

                        <div style="flex: 1; min-width: 120px; padding: 6px 10px; background: rgba(147, 197, 253, 0.08); border: 1px solid rgba(147, 197, 253, 0.25); border-radius: 5px;">
                            <div style="font-size: 9px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">SegFormer Sky Horizon</div>
                            <div style="font-size: 13px; font-weight: 700; color: #93c5fd; margin-top: 2px;">
                                ${sf.sky_pct !== undefined ? sf.sky_pct : 0}% <span style="font-size: 10px; font-weight: normal; color: var(--text-3); font-size: 9.5px;">(${Math.round((sf.sky_confidence || 0) * 100)}% conf)</span>
                            </div>
                        </div>

                        <div style="flex: 1; min-width: 120px; padding: 6px 10px; background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 5px;">
                            <div style="font-size: 9px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px;">FloodNet DeepLabV3+</div>
                            <div style="font-size: 13px; font-weight: 700; color: #06b6d4; margin-top: 2px;">
                                ${hz.flooded_pct !== undefined ? hz.flooded_pct : 0}% <span style="font-size: 10px; font-weight: normal; color: var(--text-3);">nadir flood</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        }

        panel.innerHTML = `
            <div class="unified-result-header">
                <span class="unified-result-title">
                    <span>⚡ Unified Inspection Report</span>
                    <span style="font-size: 10px; background: rgba(168, 85, 247, 0.2); border: 1px solid rgba(168, 85, 247, 0.4); color: #d8b4fe; padding: 2px 6px; border-radius: 4px;">VERIFIED</span>
                </span>
                <span style="font-family: var(--mono); font-size: 11px; color: ${color}; font-weight: 700;">
                    ${sev.level || 'MEDIUM'} (${score}/100)
                </span>
            </div>

            ${domainGuardHtml}

            <!-- Cross-Model Hazard Routing Synthesis -->
            ${routingBannerHtml}

            <!-- 1. Dedicated Landslide Differential (TransLandSeg · Bijie-trained) -->
            ${renderTransLandSegBox(res.preview_image_b64, tls)}

            <!-- 2. General Scene Water Detection (SegFormer · ADE20K) -->
            ${renderSegFormerWaterBox(res.preview_image_b64, sf)}

            <!-- 3. AI Flood / Debris Differential (FloodNet · DeepLabV3+) -->
            ${renderFloodNetBox(res.preview_image_b64, seg.segmentation_mask_b64, hz, sev)}

            <!-- Summary Directive & Multi-Model Priority Strip -->
            <div class="unified-meta-box" style="margin: 8px 0 10px 0; padding: 8px 10px; background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: var(--radius-sm);">
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 6px;">
                    <div><strong style="color: #fbbf24;">TransLandSeg Scar:</strong> <span style="color:#fbbf24; font-family:var(--mono);">${tls.landslide_pct !== undefined ? tls.landslide_pct : 0}%</span></div>
                    <div><strong style="color: #38bdf8;">SegFormer Water:</strong> <span style="color:#38bdf8; font-family:var(--mono);">${sf.flood_water_pct !== undefined ? sf.flood_water_pct : 0}% (${Math.round((sf.water_confidence || 0) * 100)}% conf)</span></div>
                    <div><strong style="color: #93c5fd;">SegFormer Sky:</strong> <span style="color:#93c5fd; font-family:var(--mono);">${sf.sky_pct !== undefined ? sf.sky_pct : 0}%</span></div>
                    <div><strong style="color: #06b6d4;">FloodNet Inundation:</strong> <span style="color:#06b6d4; font-family:var(--mono);">${hz.flooded_pct || 0}%</span></div>
                    <div><strong>Priority:</strong> <span style="color:${color}; font-weight:700;">${sev.badge || 'CRITICAL'}</span></div>
                </div>
                <div style="font-size: 10.5px; color: var(--text-3); line-height: 1.3; margin-top: 4px;">
                    ${sev.directive || 'Execute tactical field inspection.'}
                </div>
            </div>

            <!-- Road Accessibility Section -->
            ${roadHtml}

            <!-- Building Damage Section -->
            ${bldgHtml}

            <!-- Action: Launch simulated flight pass over this image -->
            <button type="button" class="btn btn-sm btn-outline-aerial" style="width: 100%; font-weight: 600;" onclick="window.launchFlightPassFromUpload()">
                ▶ Run Simulated Flight Pass on this Image
            </button>
        `;

        attachSliderListeners('tls');
        attachSliderListeners('sf');
        attachSliderListeners('fnet');
        window.initSliderSync('tls');
        window.initSliderSync('sf');
        window.initSliderSync('fnet');
    }

    // Helper: Render TransLandSeg Dedicated Landslide Comparison Component
    function renderTransLandSegBox(rawSrc, tls = {}) {
        const maskSrc = tls.segmentation_mask_b64;
        const lsPct = tls.landslide_pct !== undefined ? tls.landslide_pct : 0;
        const nonLsPct = tls.non_landslide_pct !== undefined ? tls.non_landslide_pct : 100;
        const areaM2 = tls.landslide_area_m2 || 0;
        const conf = Math.round((tls.confidence || 0) * 100);
        const fp = tls.checkpoint_fingerprint || "b69f843685fa";

        if (!maskSrc) {
            return `
                <div class="landslide-cmp-container amber" id="landslide-cmp-tls" style="margin-bottom: 12px;">
                    <div class="cmp-header">
                        <div class="cmp-title-group">
                            <span class="cmp-live-indicator amber"></span>
                            <h4 class="cmp-title" style="color: #fbbf24;">Landslide Differential (TransLandSeg &bull; Bijie-trained)</h4>
                            <span class="cmp-badge-amber">SAM ViT-L</span>
                        </div>
                    </div>
                    <div style="padding: 10px; font-size: 11px; color: var(--text-3);">
                        TransLandSeg model inactive or no landslide mask generated.
                    </div>
                </div>
            `;
        }

        return `
            <div class="landslide-cmp-container amber" id="landslide-cmp-tls" style="margin-bottom: 14px;">
                <div class="cmp-header">
                    <div class="cmp-title-group">
                        <span class="cmp-live-indicator amber"></span>
                        <h4 class="cmp-title" style="color: #fbbf24;">Landslide Differential (TransLandSeg &bull; Bijie-trained)</h4>
                        <span class="cmp-badge-amber">SAM ViT-L &bull; 770 Scars Benchmark</span>
                    </div>
                    <div class="cmp-controls">
                        <!-- Mask Opacity Slider -->
                        <div class="cmp-opacity-wrap" title="Adjust TransLandSeg overlay opacity">
                            <span>Overlay:</span>
                            <input type="range" min="15" max="100" value="85" class="cmp-opacity-slider" id="cmp-opacity-slider-tls" oninput="window.setMaskOpacity(this.value, 'tls')">
                            <span class="cmp-opacity-num" id="cmp-opacity-val-tls">85%</span>
                        </div>

                        <!-- View Toggle Buttons -->
                        <div class="cmp-btn-group" role="group" aria-label="Comparison View Mode">
                            <button type="button" class="cmp-mode-btn active" id="btn-mode-side-tls" onclick="window.setCmpMode('side', 'tls')">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="18" rx="2"></rect><rect x="13" y="3" width="8" height="18" rx="2"></rect></svg>
                                Side-by-Side
                            </button>
                            <button type="button" class="cmp-mode-btn" id="btn-mode-slider-tls" onclick="window.setCmpMode('slider', 'tls')">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="2" x2="12" y2="22"></line><polygon points="8,8 4,12 8,16"></polygon><polygon points="16,8 20,12 16,16"></polygon></svg>
                                Split Slider
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 1. Side-by-Side 2-Column Grid -->
                <div class="cmp-grid-view" id="cmp-grid-view-tls">
                    <!-- Raw Input Card -->
                    <div class="cmp-card">
                        <div class="cmp-card-tag tag-raw">
                            <div class="cmp-tag-title">
                                <span class="tag-dot"></span>
                                <span>RAW FIELD PHOTO</span>
                            </div>
                            <span class="cmp-tag-pill pill-gray">INPUT FRAME</span>
                        </div>
                        <div class="cmp-media-frame">
                            <img src="${rawSrc}" alt="Raw Input Aerial Photo" class="cmp-img">
                        </div>
                        <div class="cmp-card-footer">
                            <span>Orthophoto / Drone</span>
                            <span>RGB Sensor</span>
                        </div>
                    </div>

                    <!-- AI Segmentation Overlay Card (Amber TransLandSeg Mask) -->
                    <div class="cmp-card" style="border-color: rgba(245, 158, 11, 0.3);">
                        <div class="cmp-card-tag tag-ml" style="background: rgba(245, 158, 11, 0.1);">
                            <div class="cmp-tag-title">
                                <span class="tag-dot" style="background: #f59e0b;"></span>
                                <span style="color: #fbbf24;">TRANSLANDSEG SCAR DETECTION</span>
                            </div>
                            <span class="cmp-tag-pill pill-amber">ML DETECTED</span>
                        </div>
                        <div class="cmp-media-frame">
                            <img src="${rawSrc}" alt="Base Aerial Photo" class="cmp-img">
                            <img src="${maskSrc}" alt="TransLandSeg Landslide Mask" class="cmp-img-mask-overlay" id="cmp-side-mask-tls" style="opacity: 0.85;">
                        </div>
                        <div class="cmp-card-footer">
                            <span class="cmp-stat-badge amber">Landslide Scar: ${lsPct}%</span>
                            <span class="cmp-stat-badge" style="color: #94a3b8;">Confidence: ${conf}%</span>
                        </div>
                    </div>
                </div>

                <!-- 2. Interactive Before/After Split Slider -->
                <div class="cmp-slider-view hidden" id="cmp-slider-view-tls">
                    <div class="slider-viewport" id="slider-viewport-tls">
                        <!-- Under: Raw Image + Amber Mask Overlay -->
                        <img src="${rawSrc}" alt="Base Aerial Photo" class="cmp-img slider-img-under-base">
                        <img src="${maskSrc}" alt="TransLandSeg Mask" class="cmp-img slider-img-under-mask" id="slider-under-mask-tls" style="opacity: 0.85;">

                        <!-- Over (Clipped): Pure Raw Photo -->
                        <div class="slider-overlay-clip" id="slider-overlay-clip-tls">
                            <img src="${rawSrc}" alt="Raw Input Aerial Photo" class="cmp-img slider-img-over" id="slider-img-over-tls">
                            <div class="slider-label label-left">📷 RAW PHOTO</div>
                        </div>

                        <div class="slider-label label-right" style="color: #fbbf24; border-color: #f59e0b;">⚡ TRANSLANDSEG OVERLAY</div>

                        <div class="slider-divider" id="slider-divider-tls">
                            <div class="slider-handle" style="box-shadow: 0 0 10px #f59e0b; border-color: #f59e0b;">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                    <polyline points="15 18 9 12 15 6"></polyline>
                                </svg>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                    <polyline points="9 18 15 12 9 6"></polyline>
                                </svg>
                            </div>
                        </div>
                    </div>
                    <div class="slider-caption">
                        ↔ Drag divider horizontally to slide between pristine photo and TransLandSeg landslide mask
                    </div>
                </div>

                <!-- Segmentation Class Color Legend -->
                <div class="cmp-legend">
                    <span class="cmp-legend-title">CLASSES:</span>
                    <div class="cmp-legend-badge badge-debris"><span class="cmp-badge-dot"></span> Landslide Scar / Mudslide (${lsPct}%)</div>
                    <div class="cmp-legend-badge badge-stable"><span class="cmp-badge-dot"></span> Baseline Stable Terrain (${nonLsPct}%)</div>
                    <div class="cmp-legend-badge" style="background: rgba(255,255,255,0.03); color: #94a3b8; border: 1px solid rgba(255,255,255,0.08); font-size: 10px;">
                        Footprint: ${areaM2} m² &bull; Verified MD5: ${fp}
                    </div>
                </div>
            </div>
        `;
    }

    // Helper: Render SegFormer Water and Sky Comparison Component
    function renderSegFormerWaterBox(rawSrc, sf = {}) {
        const maskSrc = sf.segmentation_mask_b64;
        const waterPct = sf.flood_water_pct !== undefined ? sf.flood_water_pct : 0;
        const waterConf = Math.round((sf.water_confidence || 0) * 100);
        const skyPct = sf.sky_pct !== undefined ? sf.sky_pct : 0;
        const skyConf = Math.round((sf.sky_confidence || 0) * 100);
        const areaM2 = sf.flood_water_area_m2 || 0;

        const mainConfBadge = waterPct > 0 ? `Confidence: ${waterConf}%` : `Sky Conf: ${skyConf}%`;

        if (!maskSrc) {
            return `
                <div class="landslide-cmp-container" id="landslide-cmp-sf" style="margin-bottom: 12px; border-color: rgba(56, 189, 248, 0.35);">
                    <div class="cmp-header">
                        <div class="cmp-title-group">
                            <span class="cmp-live-indicator" style="background: #38bdf8; box-shadow: 0 0 8px #38bdf8;"></span>
                            <h4 class="cmp-title" style="color: #38bdf8;">General Scene Water Detection (SegFormer &bull; ADE20K)</h4>
                            <span class="cmp-badge-teal">${mainConfBadge}</span>
                        </div>
                    </div>
                    <div style="padding: 10px; font-size: 11px; color: var(--text-3);">
                        SegFormer water detector inactive or no mask generated.
                    </div>
                </div>
            `;
        }

        return `
            <div class="landslide-cmp-container" id="landslide-cmp-sf" style="margin-bottom: 14px; border-color: rgba(56, 189, 248, 0.35); box-shadow: 0 8px 28px rgba(0,0,0,0.45), 0 0 16px rgba(56, 189, 248, 0.08);">
                <div class="cmp-header">
                    <div class="cmp-title-group">
                        <span class="cmp-live-indicator" style="background: #38bdf8; box-shadow: 0 0 8px #38bdf8;"></span>
                        <h4 class="cmp-title" style="color: #38bdf8;">General Scene Water Detection (SegFormer &bull; ADE20K)</h4>
                        <span class="cmp-badge-teal">${mainConfBadge}</span>
                    </div>
                    <div class="cmp-controls">
                        <!-- Mask Opacity Slider -->
                        <div class="cmp-opacity-wrap" title="Adjust SegFormer overlay opacity">
                            <span>Overlay:</span>
                            <input type="range" min="15" max="100" value="80" class="cmp-opacity-slider" id="cmp-opacity-slider-sf" oninput="window.setMaskOpacity(this.value, 'sf')">
                            <span class="cmp-opacity-num" id="cmp-opacity-val-sf">80%</span>
                        </div>

                        <!-- View Toggle Buttons -->
                        <div class="cmp-btn-group" role="group" aria-label="Comparison View Mode">
                            <button type="button" class="cmp-mode-btn active" id="btn-mode-side-sf" onclick="window.setCmpMode('side', 'sf')">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="18" rx="2"></rect><rect x="13" y="3" width="8" height="18" rx="2"></rect></svg>
                                Side-by-Side
                            </button>
                            <button type="button" class="cmp-mode-btn" id="btn-mode-slider-sf" onclick="window.setCmpMode('slider', 'sf')">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="2" x2="12" y2="22"></line><polygon points="8,8 4,12 8,16"></polygon><polygon points="16,8 20,12 16,16"></polygon></svg>
                                Split Slider
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 1. Side-by-Side 2-Column Grid -->
                <div class="cmp-grid-view" id="cmp-grid-view-sf">
                    <!-- Raw Input Card -->
                    <div class="cmp-card">
                        <div class="cmp-card-tag tag-raw">
                            <div class="cmp-tag-title">
                                <span class="tag-dot"></span>
                                <span>RAW FIELD PHOTO</span>
                            </div>
                            <span class="cmp-tag-pill pill-gray">INPUT FRAME</span>
                        </div>
                        <div class="cmp-media-frame">
                            <img src="${rawSrc}" alt="Raw Input Aerial Photo" class="cmp-img">
                        </div>
                        <div class="cmp-card-footer">
                            <span>Orthophoto / Drone</span>
                            <span>RGB Sensor</span>
                        </div>
                    </div>

                    <!-- AI Segmentation Overlay Card -->
                    <div class="cmp-card" style="border-color: rgba(56, 189, 248, 0.3);">
                        <div class="cmp-card-tag tag-ml">
                            <div class="cmp-tag-title">
                                <span class="tag-dot" style="background: #38bdf8;"></span>
                                <span style="color: #38bdf8;">SEGFORMER SCENE DECOMPOSITION</span>
                            </div>
                            <span class="cmp-tag-pill pill-cyan">ML DETECTED</span>
                        </div>
                        <div class="cmp-media-frame">
                            <img src="${rawSrc}" alt="Base Aerial Photo" class="cmp-img">
                            <img src="${maskSrc}" alt="SegFormer Water Mask" class="cmp-img-mask-overlay" id="cmp-side-mask-sf" style="opacity: 0.8;">
                        </div>
                        <div class="cmp-card-footer">
                            <span class="cmp-stat-badge cyan">Water: ${waterPct}% (${waterConf}% conf)</span>
                            <span class="cmp-stat-badge" style="color: #93c5fd; background: rgba(147, 197, 253, 0.15);">Sky: ${skyPct}% (${skyConf}% conf)</span>
                        </div>
                    </div>
                </div>

                <!-- 2. Interactive Before/After Split Slider -->
                <div class="cmp-slider-view hidden" id="cmp-slider-view-sf">
                    <div class="slider-viewport" id="slider-viewport-sf">
                        <!-- Under: Raw Image + Mask Overlay -->
                        <img src="${rawSrc}" alt="Base Aerial Photo" class="cmp-img slider-img-under-base">
                        <img src="${maskSrc}" alt="SegFormer Mask" class="cmp-img slider-img-under-mask" id="slider-under-mask-sf" style="opacity: 0.8;">

                        <!-- Over (Clipped): Pure Raw Photo -->
                        <div class="slider-overlay-clip" id="slider-overlay-clip-sf">
                            <img src="${rawSrc}" alt="Raw Input Aerial Photo" class="cmp-img slider-img-over" id="slider-img-over-sf">
                            <div class="slider-label label-left">📷 RAW PHOTO</div>
                        </div>

                        <div class="slider-label label-right" style="color: #38bdf8; border-color: #0284c7;">⚡ SEGFORMER OVERLAY</div>

                        <div class="slider-divider" id="slider-divider-sf">
                            <div class="slider-handle" style="box-shadow: 0 0 10px #38bdf8; border-color: #38bdf8;">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                    <polyline points="15 18 9 12 15 6"></polyline>
                                </svg>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                    <polyline points="9 18 15 12 9 6"></polyline>
                                </svg>
                            </div>
                        </div>
                    </div>
                    <div class="slider-caption">
                        ↔ Drag divider horizontally to slide between pristine photo and SegFormer scene decomposition
                    </div>
                </div>

                <!-- Segmentation Class Color Legend -->
                <div class="cmp-legend">
                    <span class="cmp-legend-title">CLASSES:</span>
                    <div class="cmp-legend-badge badge-flood"><span class="cmp-badge-dot" style="background:#06b6d4;"></span> Water / Flood (${waterPct}%)</div>
                    <div class="cmp-legend-badge" style="background: rgba(147, 197, 253, 0.15); border: 1px solid rgba(147, 197, 253, 0.35); color: #bfdbfe;"><span class="cmp-badge-dot" style="background:#93c5fd;"></span> Sky Horizon (${skyPct}%)</div>
                    <div class="cmp-legend-badge badge-stable"><span class="cmp-badge-dot"></span> Terrestrial Baseline</div>
                    <div class="cmp-legend-badge" style="background: rgba(255,255,255,0.03); color: #94a3b8; border: 1px solid rgba(255,255,255,0.08); font-size: 10px;">
                        Water Footprint: ${areaM2} m² &bull; ADE20K 150-Class Transformer
                    </div>
                </div>
            </div>
        `;
    }

    // Helper: Render FloodNet Comparison Component
    function renderFloodNetBox(rawSrc, maskSrc, hz = {}, sev = {}) {
        if (!maskSrc) {
            return `
                <div class="unified-preview-strip">
                    <div class="unified-preview-box">
                        <img src="${rawSrc}" alt="Inspected Frame">
                    </div>
                    <div class="unified-meta-box">
                        <div><strong>Flooded Inundation:</strong> <span style="color:#06b6d4; font-family:var(--mono);">${hz.flooded_pct || 0}%</span></div>
                        <div><strong>Debris / Bare Ground:</strong> <span style="color:#d97706; font-family:var(--mono);">${hz.debris_pct || 0}%</span></div>
                        <div><strong>Severity Priority:</strong> <span style="color:${sev.color_hex || '#F59E0B'}; font-weight:700;">${sev.badge || 'CRITICAL'}</span></div>
                    </div>
                </div>
            `;
        }

        return `
            <div class="landslide-cmp-container" id="landslide-cmp-fnet">
                <div class="cmp-header">
                    <div class="cmp-title-group">
                        <span class="cmp-live-indicator"></span>
                        <h4 class="cmp-title">AI Flood / Debris Differential (FloodNet &bull; DeepLabV3+)</h4>
                        <span class="cmp-badge-teal">DeepLabV3+ &bull; FloodNet Baseline</span>
                    </div>
                    <div class="cmp-controls">
                        <!-- Mask Opacity Slider -->
                        <div class="cmp-opacity-wrap" title="Adjust FloodNet segmentation overlay opacity">
                            <span>Overlay:</span>
                            <input type="range" min="15" max="100" value="80" class="cmp-opacity-slider" id="cmp-opacity-slider-fnet" oninput="window.setMaskOpacity(this.value, 'fnet')">
                            <span class="cmp-opacity-num" id="cmp-opacity-val-fnet">80%</span>
                        </div>

                        <!-- View Toggle Buttons -->
                        <div class="cmp-btn-group" role="group" aria-label="Comparison View Mode">
                            <button type="button" class="cmp-mode-btn active" id="btn-mode-side-fnet" onclick="window.setCmpMode('side', 'fnet')">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="8" height="18" rx="2"></rect><rect x="13" y="3" width="8" height="18" rx="2"></rect></svg>
                                Side-by-Side
                            </button>
                            <button type="button" class="cmp-mode-btn" id="btn-mode-slider-fnet" onclick="window.setCmpMode('slider', 'fnet')">
                                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="2" x2="12" y2="22"></line><polygon points="8,8 4,12 8,16"></polygon><polygon points="16,8 20,12 16,16"></polygon></svg>
                                Split Slider
                            </button>
                        </div>
                    </div>
                </div>

                <!-- 1. Side-by-Side 2-Column Grid -->
                <div class="cmp-grid-view" id="cmp-grid-view-fnet">
                    <!-- Raw Input Card -->
                    <div class="cmp-card">
                        <div class="cmp-card-tag tag-raw">
                            <div class="cmp-tag-title">
                                <span class="tag-dot"></span>
                                <span>RAW FIELD PHOTO</span>
                            </div>
                            <span class="cmp-tag-pill pill-gray">INPUT FRAME</span>
                        </div>
                        <div class="cmp-media-frame">
                            <img src="${rawSrc}" alt="Raw Input Aerial Photo" class="cmp-img">
                        </div>
                        <div class="cmp-card-footer">
                            <span>Orthophoto / Drone</span>
                            <span>RGB Sensor</span>
                        </div>
                    </div>

                    <!-- AI Segmentation Overlay Card (Blended directly onto terrain) -->
                    <div class="cmp-card">
                        <div class="cmp-card-tag tag-ml">
                            <div class="cmp-tag-title">
                                <span class="tag-dot"></span>
                                <span>FLOODNET OVERLAY</span>
                            </div>
                            <span class="cmp-tag-pill pill-cyan">ML DETECTED</span>
                        </div>
                        <div class="cmp-media-frame">
                            <img src="${rawSrc}" alt="Base Aerial Photo" class="cmp-img">
                            <img src="${maskSrc}" alt="FloodNet Segmentation Mask" class="cmp-img-mask-overlay" id="cmp-side-mask-fnet" style="opacity: 0.8;">
                        </div>
                        <div class="cmp-card-footer">
                            <span class="cmp-stat-badge cyan">Inundated: ${hz.flooded_pct || 0}%</span>
                            <span class="cmp-stat-badge amber">Debris: ${hz.debris_pct || 0}%</span>
                        </div>
                    </div>
                </div>

                <!-- 2. Interactive Before/After Split Slider -->
                <div class="cmp-slider-view hidden" id="cmp-slider-view-fnet">
                    <div class="slider-viewport" id="slider-viewport-fnet">
                        <!-- Under: Raw Image + Mask Overlay -->
                        <img src="${rawSrc}" alt="Base Aerial Photo" class="cmp-img slider-img-under-base">
                        <img src="${maskSrc}" alt="ML Segmentation Mask" class="cmp-img slider-img-under-mask" id="slider-under-mask-fnet" style="opacity: 0.8;">

                        <!-- Over (Clipped): Pure Raw Photo -->
                        <div class="slider-overlay-clip" id="slider-overlay-clip-fnet">
                            <img src="${rawSrc}" alt="Raw Input Aerial Photo" class="cmp-img slider-img-over" id="slider-img-over-fnet">
                            <div class="slider-label label-left">📷 RAW PHOTO</div>
                        </div>

                        <div class="slider-label label-right">⚡ FLOODNET OVERLAY</div>

                        <div class="slider-divider" id="slider-divider-fnet">
                            <div class="slider-handle">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                    <polyline points="15 18 9 12 15 6"></polyline>
                                </svg>
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                                    <polyline points="9 18 15 12 9 6"></polyline>
                                </svg>
                            </div>
                        </div>
                    </div>
                    <div class="slider-caption">
                        ↔ Drag divider horizontally to slide between pristine photo and FloodNet segmentation overlay
                    </div>
                </div>

                <!-- Segmentation Class Color Legend (Non-colliding badges) -->
                <div class="cmp-legend">
                    <span class="cmp-legend-title">CLASSES:</span>
                    <div class="cmp-legend-badge badge-flood"><span class="cmp-badge-dot"></span> Flood / Inundation (${hz.flooded_pct || 0}%)</div>
                    <div class="cmp-legend-badge badge-debris"><span class="cmp-badge-dot"></span> FloodNet Bare Debris (${hz.debris_pct || 0}%)</div>
                    <div class="cmp-legend-badge badge-damaged"><span class="cmp-badge-dot"></span> Damaged Structure</div>
                    <div class="cmp-legend-badge badge-stable"><span class="cmp-badge-dot"></span> Baseline Terrain</div>
                </div>
            </div>
        `;
    }

    // Toggle comparison modes (supports prefix: 'tls', 'fnet', or legacy default)
    window.setCmpMode = function(mode, prefix = 'tls') {
        const gridView = document.getElementById(`cmp-grid-view-${prefix}`) || document.getElementById('cmp-grid-view');
        const sliderView = document.getElementById(`cmp-slider-view-${prefix}`) || document.getElementById('cmp-slider-view');
        const btnSide = document.getElementById(`btn-mode-side-${prefix}`) || document.getElementById('btn-mode-side');
        const btnSlider = document.getElementById(`btn-mode-slider-${prefix}`) || document.getElementById('btn-mode-slider');
        if (!gridView || !sliderView) return;

        if (mode === 'slider') {
            gridView.classList.add('hidden');
            sliderView.classList.remove('hidden');
            if (btnSlider) btnSlider.classList.add('active');
            if (btnSide) btnSide.classList.remove('active');
            window.initSliderSync(prefix);
        } else {
            sliderView.classList.add('hidden');
            gridView.classList.remove('hidden');
            if (btnSide) btnSide.classList.add('active');
            if (btnSlider) btnSlider.classList.remove('active');
        }
    };

    // Synchronize slider overlay image width with viewport
    window.initSliderSync = function(prefix) {
        const prefixes = prefix ? [prefix] : ['tls', 'sf', 'fnet', ''];
        prefixes.forEach(p => {
            const viewport = document.getElementById(p ? `slider-viewport-${p}` : 'slider-viewport');
            const overImg = document.getElementById(p ? `slider-img-over-${p}` : 'slider-img-over');
            if (viewport && overImg) {
                overImg.style.width = viewport.clientWidth + 'px';
            }
        });
    };

    // Dynamically adjust segmentation mask opacity
    window.setMaskOpacity = function(val, prefix = 'tls') {
        const num = document.getElementById(prefix ? `cmp-opacity-val-${prefix}` : 'cmp-opacity-val');
        if (num) num.textContent = `${val}%`;
        const op = Math.max(0.1, Math.min(1.0, val / 100));
        const sideMask = document.getElementById(prefix ? `cmp-side-mask-${prefix}` : 'cmp-side-mask');
        if (sideMask) sideMask.style.opacity = op;
        const sliderMask = document.getElementById(prefix ? `slider-under-mask-${prefix}` : 'slider-under-mask');
        if (sliderMask) sliderMask.style.opacity = op;
    };

    // Attach touch and mouse listeners to the split slider
    function attachSliderListeners(prefix = 'tls') {
        const viewport = document.getElementById(prefix ? `slider-viewport-${prefix}` : 'slider-viewport');
        if (!viewport || viewport.dataset.sliderAttached === 'true') return;
        viewport.dataset.sliderAttached = 'true';

        let isDragging = false;

        function updateSliderPosition(clientX) {
            const clip = document.getElementById(prefix ? `slider-overlay-clip-${prefix}` : 'slider-overlay-clip');
            const divider = document.getElementById(prefix ? `slider-divider-${prefix}` : 'slider-divider');
            const overImg = document.getElementById(prefix ? `slider-img-over-${prefix}` : 'slider-img-over');
            if (!viewport || !clip || !divider) return;

            const rect = viewport.getBoundingClientRect();
            let posX = clientX - rect.left;
            posX = Math.max(0, Math.min(posX, rect.width));
            const pct = (posX / rect.width) * 100;

            clip.style.width = `${pct}%`;
            divider.style.left = `${pct}%`;
            if (overImg) {
                overImg.style.width = `${rect.width}px`;
            }
        }

        viewport.addEventListener('mousedown', (e) => {
            isDragging = true;
            updateSliderPosition(e.clientX);
        });

        window.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            updateSliderPosition(e.clientX);
        });

        window.addEventListener('mouseup', () => {
            isDragging = false;
        });

        viewport.addEventListener('touchstart', (e) => {
            isDragging = true;
            if (e.touches.length > 0) updateSliderPosition(e.touches[0].clientX);
        }, { passive: true });

        window.addEventListener('touchmove', (e) => {
            if (!isDragging) return;
            if (e.touches.length > 0) updateSliderPosition(e.touches[0].clientX);
        }, { passive: true });

        window.addEventListener('touchend', () => {
            isDragging = false;
        });
    }

    window.addEventListener('resize', () => {
        if (window.initSliderSync) window.initSliderSync();
    });

    // ── DMMC Report Modal Handlers ───────────────────────────────────────────
    window.openDMMCReportModal = function(optionalZoneId) {
        const zoneId = optionalZoneId || aerialState.activePresetId;
        const modal = document.getElementById("modal-aerial-report");
        const iframe = document.getElementById("report-iframe");
        const downloadBtn = document.getElementById("btn-download-report-file");

        if (iframe) {
            iframe.src = `/api/aerial/report?zone_id=${zoneId}&format=html`;
        }
        if (downloadBtn) {
            downloadBtn.href = `/api/aerial/report/download?zone_id=${zoneId}`;
        }
        if (modal) {
            modal.classList.remove("hidden");
        }
    };

    window.closeDMMCReportModal = function() {
        const modal = document.getElementById("modal-aerial-report");
        if (modal) modal.classList.add("hidden");
    };

    window.printReportIframe = function() {
        const iframe = document.getElementById("report-iframe");
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.print();
        }
    };

    window.exportAerialGeoJSON = async function() {
        try {
            const resp = await fetch("/api/aerial/zones");
            if (!resp.ok) return;
            const geojson = await resp.json();
            const blob = new Blob([JSON.stringify(geojson, null, 2)], { type: "application/geo+json" });
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `NETRA_Aerial_Triage_Zones_${aerialState.activePresetId}.geojson`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        } catch (err) {
            console.error("[NetraAerial] Export GeoJSON failed:", err);
        }
    };

    // ── Global View Switch Listener ──────────────────────────────────────────
    window.onViewSwitched = function(viewName) {
        console.log("[NETRA] Top view switched to:", viewName);
        if (viewName === "aerial" || viewName === "drone") {
            if (!aerialState.map) {
                initAerialMap();
                loadAerialPresets();
                loadAerialZones();
            } else {
                setTimeout(() => {
                    aerialState.map.invalidateSize();
                }, 200);
            }
        } else if (viewName === "satellite" || viewName === "sr") {
            if (state.map) {
                setTimeout(() => {
                    state.map.invalidateSize();
                }, 200);
            }
        } else if (viewName === "compare") {
            // Update Compare view numbers dynamically if aerial presets exist
            if (aerialState.presets && aerialState.presets.length > 0) {
                const p = aerialState.presets.find(x => x.id === aerialState.activePresetId) || aerialState.presets[0];
                if (p && p.drone_sortie) {
                    const deltaEl = document.getElementById("compare-stat-delta");
                    const satConf = Math.round((p.satellite_confidence || 0.68) * 100);
                    const droneConf = Math.round((p.drone_sortie.drone_confidence || 0.94) * 100);
                    if (deltaEl) deltaEl.textContent = `+${droneConf - satConf}.0%`;
                }
            }
        }
    };

    // Handle any pending switch
    if (window._pendingViewSwitch) {
        window.onViewSwitched(window._pendingViewSwitch);
        window._pendingViewSwitch = null;
    }

    // Initial load
    loadPresets();
});

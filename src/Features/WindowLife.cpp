// Window Life - runtime implementation.
// Ported to Community Shaders from PIXL Renderer (https://github.com/pixlmusic/PIXL-Renderer).
// Copyright (C) 2026 PIXL Studio
// SPDX-License-Identifier: GPL-3.0-or-later
// Additional permissions are described in the repository EXCEPTIONS.md.

#include "WindowLife.h"

#include "Globals.h"
#include "State.h"
#include "Util.h"
#include "Utils/Game.h"
#include "Utils/UI.h"

#include <DirectXTex.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <string_view>
#include <vector>

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
	WindowLife::Settings,
	EnableWindowLife,
	EnableArchitecturalGlass,
	DayShadowStrength,
	NightShadowStrength,
	DayActivity,
	EveningActivity,
	LateNightActivity,
	ParallaxDepth,
	InteriorParallaxDepth,
	EnableOutdoorViews,
	OutdoorViewStrength,
	OutdoorViewEmission,
	Refraction,
	SilhouetteSoftness,
	HumanScale,
	OccupantOpacity,
	CurtainStrength,
	RoomDepthStrength,
	EnableAuthoredRooms,
	AuthoredRoomStrength,
	InteriorContrast,
	InteriorEmission,
	InteriorScale,
	InteriorSoftness,
	RoomVolumeStrength,
	WindowRecess,
	AutomaticRoomSizing,
	UseExactGlassMasks,
	GroupMultiPaneWindows,
	MultiWindowMeshScale,
	EnableInteriorPassers,
	DistanceFadeStart,
	DistanceFadeEnd,
	PaneThreshold,
	PaneSoftness,
	RoomWidth,
	RoomHeight,
	MotionSpeed,
	GlassStrength,
	GlassReflectionBoost,
	GlassRoughness,
	GlassTransmission,
	GlassDirtStrength,
	GlassDistortion,
	GlassNormalRetention,
	EnvironmentReflectionStrength,
	InteriorLightingResponse,
	WeatherGlassResponse,
	SunGlintStrength,
	DirectionalRevealStrength,
	RoomVariationStrength,
	CloseLayerFeather,
	SuppressWindowAutoPOM,
	MinShallowWindowRadius,
	MinFullWindowRadius,
	FullWindowVerticality,
	DebugWindowDetection,
	DebugRoomProjection);

namespace
{
	// ---- Glass-mask pane bounds ----------------------------------------------------
	// Window masks are often atlases: every pane is its own island of white in
	// the texture, separated by black gutters. The shader's own search walks a
	// coarse mip and cannot separate those islands, so the room fit fails. On
	// load we label each island once and store its UV rectangle in a small
	// lookup texture (R16G16B16A16: minU, minV, maxU, maxV). The shader reads
	// the rectangle under the current pixel and frames the room to that pane.
	struct PaneBoundsResult
	{
		std::size_t width = 0;
		std::size_t height = 0;
		std::vector<std::uint16_t> texels;  // 4 per texel
		int panes = 0;
	};

	PaneBoundsResult BuildPaneBounds(const std::uint8_t* rgba, std::size_t rowPitch, std::size_t width, std::size_t height)
	{
		PaneBoundsResult result{};
		if (!rgba || width < 8u || height < 8u)
			return result;

		// Work on a copy of at most 512 texels on the longest side.
		const std::size_t longestSide = std::max(width, height);
		const std::size_t factor = std::max<std::size_t>(1u, longestSide / 512u);
		const int w = static_cast<int>(width / factor);
		const int h = static_cast<int>(height / factor);
		if (w < 8 || h < 8)
			return result;
		const auto at = [w](int x, int y) { return static_cast<std::size_t>(y) * static_cast<std::size_t>(w) + static_cast<std::size_t>(x); };

		std::vector<std::uint8_t> glass(static_cast<std::size_t>(w) * static_cast<std::size_t>(h), 0u);
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				float sum = 0.0f;
				for (std::size_t sy = 0; sy < factor; ++sy) {
					const std::uint8_t* row = rgba + (static_cast<std::size_t>(y) * factor + sy) * rowPitch;
					for (std::size_t sx = 0; sx < factor; ++sx) {
						const std::uint8_t* px = row + (static_cast<std::size_t>(x) * factor + sx) * 4u;
						sum += 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
					}
				}
				glass[at(x, y)] = sum / (255.0f * static_cast<float>(factor * factor)) > 0.5f ? 1u : 0u;
			}
		}

		// Close small holes and frost specks so each pane is one island. The
		// radius bridges gaps of up to 4 texels, well below a typical gutter.
		const int radius = std::max(1, std::max(w, h) / 256);
		const auto morph = [&](const std::vector<std::uint8_t>& src, bool dilate) {
			std::vector<std::uint8_t> horizontal(src.size());
			std::vector<std::uint8_t> out(src.size());
			for (int y = 0; y < h; ++y) {
				for (int x = 0; x < w; ++x) {
					std::uint8_t v = dilate ? 0u : 1u;
					for (int d = -radius; d <= radius; ++d) {
						const int xx = x + d;
						const std::uint8_t s = (xx >= 0 && xx < w) ? src[at(xx, y)] : (dilate ? 0u : 1u);
						v = dilate ? static_cast<std::uint8_t>(v | s) : static_cast<std::uint8_t>(v & s);
					}
					horizontal[at(x, y)] = v;
				}
			}
			for (int y = 0; y < h; ++y) {
				for (int x = 0; x < w; ++x) {
					std::uint8_t v = dilate ? 0u : 1u;
					for (int d = -radius; d <= radius; ++d) {
						const int yy = y + d;
						const std::uint8_t s = (yy >= 0 && yy < h) ? horizontal[at(x, yy)] : (dilate ? 0u : 1u);
						v = dilate ? static_cast<std::uint8_t>(v | s) : static_cast<std::uint8_t>(v & s);
					}
					out[at(x, y)] = v;
				}
			}
			return out;
		};
		const std::vector<std::uint8_t> closed = morph(morph(glass, true), false);

		// Label 4-connected islands and record their rectangles.
		struct Island
		{
			int minX, minY, maxX, maxY;
			std::size_t area;
		};
		std::vector<Island> islands;
		std::vector<int> label(closed.size(), -1);
		std::vector<int> stack;
		for (int y = 0; y < h; ++y) {
			for (int x = 0; x < w; ++x) {
				if (!closed[at(x, y)] || label[at(x, y)] >= 0)
					continue;
				const int id = static_cast<int>(islands.size());
				Island island{ x, y, x, y, 0u };
				label[at(x, y)] = id;
				stack.assign(1, static_cast<int>(at(x, y)));
				while (!stack.empty()) {
					const int cell = stack.back();
					stack.pop_back();
					const int cx = cell % w;
					const int cy = cell / w;
					++island.area;
					island.minX = std::min(island.minX, cx);
					island.maxX = std::max(island.maxX, cx);
					island.minY = std::min(island.minY, cy);
					island.maxY = std::max(island.maxY, cy);
					const int neighbours[4][2] = { { cx - 1, cy }, { cx + 1, cy }, { cx, cy - 1 }, { cx, cy + 1 } };
					for (const auto& n : neighbours) {
						if (n[0] < 0 || n[1] < 0 || n[0] >= w || n[1] >= h)
							continue;
						const std::size_t next = at(n[0], n[1]);
						if (!closed[next] || label[next] >= 0)
							continue;
						label[next] = id;
						stack.push_back(static_cast<int>(next));
					}
				}
				islands.push_back(island);
			}
		}

		// Drop specks, then let the remaining panes claim nearby texels so
		// frost on a pane's edge still finds that pane.
		const std::size_t minimumArea = std::max<std::size_t>(16u, static_cast<std::size_t>(w) * static_cast<std::size_t>(h) / 2000u);
		std::vector<std::uint8_t> kept(islands.size(), 0u);
		for (std::size_t i = 0; i < islands.size(); ++i) {
			kept[i] = islands[i].area >= minimumArea ? 1u : 0u;
			result.panes += kept[i];
		}
		for (auto& l : label) {
			if (l >= 0 && !kept[static_cast<std::size_t>(l)])
				l = -1;
		}
		for (int pass = 0; pass < radius + 2; ++pass) {
			std::vector<int> grown = label;
			for (int y = 0; y < h; ++y) {
				for (int x = 0; x < w; ++x) {
					if (label[at(x, y)] >= 0)
						continue;
					const int neighbours[4][2] = { { x - 1, y }, { x + 1, y }, { x, y - 1 }, { x, y + 1 } };
					for (const auto& n : neighbours) {
						if (n[0] >= 0 && n[1] >= 0 && n[0] < w && n[1] < h && label[at(n[0], n[1])] >= 0) {
							grown[at(x, y)] = label[at(n[0], n[1])];
							break;
						}
					}
				}
			}
			label.swap(grown);
		}

		// Store at half the working size; each texel holds its pane's rectangle.
		result.width = static_cast<std::size_t>(std::max(8, w / 2));
		result.height = static_cast<std::size_t>(std::max(8, h / 2));
		result.texels.assign(result.width * result.height * 4u, 0u);
		const auto unorm = [](float v) { return static_cast<std::uint16_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 65535.0f)); };
		for (std::size_t y = 0; y < result.height; ++y) {
			for (std::size_t x = 0; x < result.width; ++x) {
				const int sx = std::min(w - 1, static_cast<int>((x * static_cast<std::size_t>(w)) / result.width + static_cast<std::size_t>(w) / (2u * result.width)));
				const int sy = std::min(h - 1, static_cast<int>((y * static_cast<std::size_t>(h)) / result.height + static_cast<std::size_t>(h) / (2u * result.height)));
				const int id = label[at(sx, sy)];
				if (id < 0)
					continue;
				const Island& island = islands[static_cast<std::size_t>(id)];
				std::uint16_t* texel = &result.texels[(y * result.width + x) * 4u];
				texel[0] = unorm(static_cast<float>(island.minX) / static_cast<float>(w));
				texel[1] = unorm(static_cast<float>(island.minY) / static_cast<float>(h));
				texel[2] = unorm(static_cast<float>(island.maxX + 1) / static_cast<float>(w));
				texel[3] = unorm(static_cast<float>(island.maxY + 1) / static_cast<float>(h));
			}
		}
		return result;
	}

	// Smooth daylight factor in [0.05, 1] from the active climate's dawn/dusk
	// timing (hours). Carried over from PIXL's AtmosphereWeather helper so this
	// feature has no dependency outside Community Shaders.
	float DaylightScale(float hour, float dawnBegin, float dawnEnd, float duskBegin, float duskEnd)
	{
		if (!std::isfinite(hour) || !std::isfinite(dawnBegin) || !std::isfinite(dawnEnd) ||
			!std::isfinite(duskBegin) || !std::isfinite(duskEnd))
			return 1.0f;
		auto wrap = [](float value) { return value - 24.0f * std::floor(value / 24.0f); };
		const float dawn = wrap(dawnEnd - dawnBegin);
		const float dusk = wrap(duskBegin - dawnBegin);
		const float night = wrap(duskEnd - dawnBegin);
		if (!(dawn > 0.0f && dawn < dusk && dusk < night))
			return 1.0f;
		const float t = wrap(hour - dawnBegin);
		auto smooth = [](float x) { x = std::clamp(x, 0.0f, 1.0f); return x * x * (3.0f - 2.0f * x); };
		const float daylight = t < dawn  ? smooth(t / dawn) :
		                       t < dusk  ? 1.0f :
		                       t < night ? 1.0f - smooth((t - dusk) / (night - dusk)) :
		                                   0.0f;
		return 0.05f + 0.95f * daylight;
	}

	// Blended snow-weather amount in [0, 1]: follows the engine's own weather
	// transition so glass frost does not pop when a storm starts or clears.
	float GetSnowWeatherAmount()
	{
		const auto* sky = globals::game::sky;
		if (!sky)
			return 0.0f;
		auto isSnow = [](const RE::TESWeather* weather) {
			return weather && weather->data.flags.any(RE::TESWeather::WeatherDataFlag::kSnow);
		};
		const float pct = std::isfinite(sky->currentWeatherPct) ? std::clamp(sky->currentWeatherPct, 0.0f, 1.0f) : 1.0f;
		const float current = isSnow(sky->currentWeather) ? 1.0f : 0.0f;
		const float last = isSnow(sky->lastWeather) ? 1.0f : 0.0f;
		return sky->lastWeather ? std::lerp(last, current, pct) : current;
	}

	std::string Lower(std::string_view value)
	{
		std::string result(value);
		std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return result;
	}

	bool ContainsAny(std::string_view text, std::initializer_list<std::string_view> tokens)
	{
		for (auto token : tokens) {
			if (text.find(token) != std::string_view::npos)
				return true;
		}
		return false;
	}

	bool EndsWith(std::string_view text, std::string_view suffix)
	{
		return text.size() >= suffix.size() &&
		       text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
	}

	std::string CanonicalWindowMaskKey(std::string_view path)
	{
		const auto slash = path.find_last_of("\\/");
		std::string key = Lower(slash == std::string_view::npos ? path : path.substr(slash + 1));
		const auto extension = key.find_last_of('.');
		if (extension != std::string::npos)
			key.resize(extension);
		// Texture replacers use both foo_mask_noalpha and foo_noalpha_mask.
		// Peel known postfixes repeatedly rather than depending on their order.
		for (int pass = 0; pass < 3; ++pass) {
			if (EndsWith(key, "_mask"))
				key.resize(key.size() - 5u);
			if (EndsWith(key, "noalpha"))
				key.resize(key.size() - 7u);
			while (!key.empty() && (key.back() == '_' || key.back() == '-'))
				key.pop_back();
		}
		return key;
	}

	float Smooth01(float x)
	{
		x = std::clamp(x, 0.0f, 1.0f);
		return x * x * (3.0f - 2.0f * x);
	}

	float SmoothRange(float edge0, float edge1, float x)
	{
		if (edge1 <= edge0)
			return x >= edge1 ? 1.0f : 0.0f;
		return Smooth01((x - edge0) / (edge1 - edge0));
	}

	std::uint32_t StableHash32(std::string_view text, std::uint32_t seed = 2166136261u)
	{
		std::uint32_t hash = seed;
		for (const unsigned char value : text) {
			hash ^= value;
			hash *= 16777619u;
		}
		return hash;
	}

	void HashCombine32(std::uint32_t& seed, std::uint32_t value)
	{
		seed ^= value + 0x9E3779B9u + (seed << 6u) + (seed >> 2u);
		seed *= 16777619u;
	}

	struct SharedWindowBounds
	{
		RE::NiPoint3 center{};
		float radius = 0.0f;
		bool grouped = false;
		std::uint32_t groupIdentity = 0u;
	};

	template <class Bound>
	bool ValidWindowBounds(const Bound& bound)
	{
		return std::isfinite(bound.radius) && bound.radius > 0.0f &&
		       std::isfinite(bound.center.x) && std::isfinite(bound.center.y) &&
		       std::isfinite(bound.center.z) &&
		       std::abs(bound.center.x) <= 1.0e8f &&
		       std::abs(bound.center.y) <= 1.0e8f &&
		       std::abs(bound.center.z) <= 1.0e8f;
	}

	SharedWindowBounds ResolveSharedWindowBounds(const RE::BSGeometry* geometry, bool enabled)
	{
		SharedWindowBounds result{};
		if (!geometry)
			return result;
		result.center = geometry->worldBound.center;
		result.radius = std::max(geometry->worldBound.radius, 0.0f);
		if (!enabled || !geometry->parent || !ValidWindowBounds(geometry->parent->worldBound))
			return result;

		const auto* parent = geometry->parent;
		const std::string name = Lower(parent->name.c_str() ? parent->name.c_str() : "");
		const bool namedWindow = ContainsAny(name, { "window", "glazing", "glass", "pane", "fenster" });
		const bool buildingOrFacade = ContainsAny(name, { "facade", "building", "architecture", "exterior", "interior", "wall", "house", "roof", "cell" });
		const float parentRadius = parent->worldBound.radius;
		// A facade root can contain several independent windows. Only a
		// modest, explicitly named aperture parent is a reliable shared owner.
		// Large parents retain the existing UV/world-space room grid.
		if (!namedWindow || buildingOrFacade || parentRadius <= result.radius * 1.025f ||
			parentRadius > 480.0f ||
			parentRadius > std::max(result.radius * 9.0f, 110.0f))
			return result;

		const float dx = parent->worldBound.center.x - result.center.x;
		const float dy = parent->worldBound.center.y - result.center.y;
		const float dz = parent->worldBound.center.z - result.center.z;
		if (dx * dx + dy * dy + dz * dz > parentRadius * parentRadius)
			return result;
		result.center = parent->worldBound.center;
		result.radius = parentRadius;
		result.grouped = true;
		result.groupIdentity = StableHash32(name);
		return result;
	}
}

void WindowLife::DrawSettings()
{
	ImGui::TextWrapped("WindowLife includes its room artwork; no separate parallax-window mod is required. It applies to recognized architectural glass. Replacement materials with unrecognized texture names may need compatibility work; boarded windows and shadow-mask helper meshes are excluded.");
	ImGui::Checkbox(T("feature.window_life.enable", "Enable Window Life"), &settings.EnableWindowLife);
	if (auto _tt = Util::HoverTooltipWrapper()) {
		ImGui::TextWrapped("%s", T("feature.window_life.enable_tooltip", "Upgrades architectural glass and adds subtle moving occupants behind suitable exterior and interior windows."));
	}

	ImGui::Checkbox(T("feature.window_life.glass_enable", "Architectural Glass Optics"), &settings.EnableArchitecturalGlass);
	if (auto _tt = Util::HoverTooltipWrapper()) {
		ImGui::TextWrapped("%s", T("feature.window_life.glass_enable_tooltip", "Applies old-glass Fresnel response, restrained waviness and stable grime/roughness variation to detected architectural panes."));
	}

	if (ImGui::CollapsingHeader(T("feature.window_life.glass", "Architectural Glass"), ImGuiTreeNodeFlags_DefaultOpen)) {
		ImGui::BeginDisabled(!settings.EnableArchitecturalGlass);
		ImGui::SliderFloat(T("feature.window_life.glass_strength", "Glass Strength"), &settings.GlassStrength, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.glass_reflection", "Reflection Response"), &settings.GlassReflectionBoost, 0.0f, 1.5f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.glass_roughness", "Base Glass Roughness"), &settings.GlassRoughness, 0.06f, 0.80f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.glass_transmission", "Glass Transmission"), &settings.GlassTransmission, 0.65f, 1.0f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.glass_dirt", "Grime Variation"), &settings.GlassDirtStrength, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.glass_distortion", "Old Glass Waviness"), &settings.GlassDistortion, 0.0f, 0.09f, "%.3f");
		ImGui::SliderFloat(T("feature.window_life.glass_normal", "Texture Normal Retention"), &settings.GlassNormalRetention, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.environment_reflection", "Environment Reflection"), &settings.EnvironmentReflectionStrength, 0.0f, 1.5f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat(T("feature.window_life.weather_glass", "Weathered Glass Response"), &settings.WeatherGlassResponse, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("Controls mip-filtered grime, rain streaks and cold-weather haze. The response uses the live precipitation state from Wetness Effects and remains attached to the glass in world space.");
		}
		ImGui::EndDisabled();
		ImGui::Checkbox(T("feature.window_life.suppress_autopom", "Keep Parallax Off Glass Panes"), &settings.SuppressWindowAutoPOM);
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("%s", T("feature.window_life.suppress_autopom_tooltip", "Suppresses authored, complex-material and synthetic parallax on detected pane pixels. Frames and surrounding architecture retain their material depth."));
		}
	}

	if (ImGui::CollapsingHeader(T("feature.window_life.visibility", "Occupancy"))) {
		ImGui::SliderFloat(T("feature.window_life.occupant_opacity", "Occupant Opacity"), &settings.OccupantOpacity, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("Controls the true opacity of the softly filtered authored people. It no longer makes them into translucent black ghosts.");
		}
		ImGui::SliderFloat(T("feature.window_life.day_activity", "Day Activity"), &settings.DayActivity, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.evening_activity", "Evening Activity"), &settings.EveningActivity, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.late_activity", "Late Night Activity"), &settings.LateNightActivity, 0.0f, 1.0f, "%.2f");
	}

	if (ImGui::CollapsingHeader(T("feature.window_life.eligibility", "Window Eligibility"))) {
		ImGui::SliderFloat(T("feature.window_life.min_shallow_radius", "Minimum Glass-Only Window Size"), &settings.MinShallowWindowRadius, 4.0f, 96.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat(T("feature.window_life.min_full_radius", "Minimum Occupied Window Size"), &settings.MinFullWindowRadius, 12.0f, 160.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
		settings.MinFullWindowRadius = std::max(settings.MinFullWindowRadius, settings.MinShallowWindowRadius + 1.0f);
		ImGui::SliderFloat(T("feature.window_life.full_verticality", "Full Occupancy Verticality"), &settings.FullWindowVerticality, 0.25f, 0.95f, "%.2f");
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("%s", T("feature.window_life.eligibility_tooltip", "Tiny dedicated meshes become glass-only. Sloped roof/awning panes are downgraded in the shader even when their parent geometry is large."));
		}
	}

	if (ImGui::CollapsingHeader(T("feature.window_life.optics", "Interior Depth"))) {
		ImGui::SliderFloat("Exterior View: Room Depth", &settings.ParallaxDepth, 0.0f, 216.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::Checkbox("Interior View: Outdoor Backgrounds", &settings.EnableOutdoorViews);
		ImGui::SliderFloat("Interior View: Parallax Depth", &settings.InteriorParallaxDepth, 0.0f, 240.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
		Util::AddTooltip("Depth of the outdoor scene and passers seen from inside. Higher values reveal more movement behind the glass as you move; shader cost is unchanged.");
		ImGui::SliderFloat("Outdoor Background Visibility", &settings.OutdoorViewStrength, 0.0f, 2.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		Util::AddTooltip("Blends authored northern landscapes behind interior windows. Requires OutdoorAtlas; missing artwork retains the original glass and passers.");
		ImGui::SliderFloat("Interior View: Background Emission", &settings.OutdoorViewEmission, 0.0f, 8.0f, "%.2fx", ImGuiSliderFlags_AlwaysClamp);
		Util::AddTooltip("Brightens the day/night landscape seen from inside through refracted glass. Does not change exterior room images, opacity or parallax. No extra texture samples.");
		ImGui::SliderFloat(T("feature.window_life.refraction", "Interior Refraction"), &settings.Refraction, 0.0f, 8.0f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.softness", "Silhouette Softness"), &settings.SilhouetteSoftness, 0.015f, 0.16f, "%.3f");
		Util::AddTooltip("Softens authored people behind the pane by selecting a filtered atlas mip and widening the alpha edge. The frame and glass stay sharp; there are no extra texture samples.");
		ImGui::SliderFloat(T("feature.window_life.human_scale", "Human Scale"), &settings.HumanScale, 0.65f, 1.35f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.curtain_strength", "Curtain Opacity"), &settings.CurtainStrength, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat(T("feature.window_life.room_depth_strength", "Recessed Room Depth"), &settings.RoomDepthStrength, 0.0f, 0.40f, "%.2f");
		ImGui::Checkbox(T("feature.window_life.authored_rooms", "Authored Room Backgrounds"), &settings.EnableAuthoredRooms);
		ImGui::SliderFloat(T("feature.window_life.authored_room_strength", "Authored Room Visibility"), &settings.AuthoredRoomStrength, 0.0f, 1.0f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.interior_contrast", "Interior Contrast"), &settings.InteriorContrast, 0.50f, 2.0f, "%.2f");
		ImGui::SliderFloat("Exterior View: Room Emission", &settings.InteriorEmission, 0.0f, 6.0f, "%.2fx");
		Util::AddTooltip("Brightens the authored room behind exterior glass while preserving its furniture and light contrast. Higher values increase HDR radiance without adding a render pass.");
		ImGui::SliderFloat(T("feature.window_life.interior_scale", "Interior Scale"), &settings.InteriorScale, 1.0f, 2.50f, "%.2fx", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Authored Room Softness", &settings.InteriorSoftness, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("Adds a controlled optical softness to the authored room artwork behind the glass. Panes, mullions and the glass surface stay sharp.");
		}
		ImGui::SliderFloat("Room Volume", &settings.RoomVolumeStrength, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		Util::AddTooltip("Projects the authored room onto a bounded rear wall, side walls, floor and ceiling as the view changes. Zero keeps the original flat mapping. No extra texture sample.");
		ImGui::SliderFloat("Window Recess", &settings.WindowRecess, 0.0f, 18.0f, "%.1f", ImGuiSliderFlags_AlwaysClamp);
		Util::AddTooltip("Places the room behind the glass and wall opening. Higher values reveal more depth at oblique angles.");
		ImGui::SliderFloat(T("feature.window_life.interior_lighting_response", "Interior Lighting Response"), &settings.InteriorLightingResponse, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("Interior Scale crops or expands the room artwork in both manual and automatic sizing modes without changing the detected glass boundary or room identity. Contrast separates furniture and walls; emission controls readability through the original glass.");
		}
		ImGui::Checkbox(T("feature.window_life.auto_room_sizing", "Automatic Room Sizing"), &settings.AutomaticRoomSizing);
		ImGui::Checkbox("Share Rooms Across Window Panes", &settings.GroupMultiPaneWindows);
		Util::AddTooltip("Multiple glass meshes under one bounded window node share a stable room, curtain selection and occupant timeline. Large facade/house nodes are excluded so separate windows remain independent.");
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("Optional texture-based aperture fitting. Manual sizing is the default. Automatic mode accepts only confident, plausibly sized fits; texture atlases can still need manual Room Width/Height. Rooms, curtains and occupants share the fit.");
		}
		ImGui::Checkbox(T("feature.window_life.interior_passers", "Interior View Passers-by"), &settings.EnableInteriorPassers);
	}

	if (ImGui::CollapsingHeader(T("feature.window_life.masking", "Pane Mask & Distance"))) {
		if (ImGui::Checkbox(T("feature.window_life.exact_glass_masks", "Use Installed Glass Masks"), &settings.UseExactGlassMasks)) {
			classificationCache.clear();
			activeDataValid = false;
		}
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("Uses diffuse-matched masks as accurate glass/frame stencils. They can also guide room bounds if native glow geometry is missing, but must pass the same strict geometry validation. Unmatched materials use conservative procedural detection.");
		}
		ImGui::SliderFloat(T("feature.window_life.pane_threshold", "Pane Threshold"), &settings.PaneThreshold, 0.0f, 0.55f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.pane_softness", "Pane Mask Softness"), &settings.PaneSoftness, 0.03f, 0.50f, "%.2f");
		Util::AddTooltip("Feathers authored glow/exact-mask panes as well as procedural panes. Frame and stone erosion guards stay active; higher values soften the transition, not the room artwork.");
		ImGui::SliderFloat(T("feature.window_life.fade_start", "Distance Fade Start"), &settings.DistanceFadeStart, 256.0f, 10000.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat(T("feature.window_life.fade_end", "Distance Fade End"), &settings.DistanceFadeEnd, 512.0f, 16000.0f, "%.0f", ImGuiSliderFlags_AlwaysClamp);
		settings.DistanceFadeEnd = std::max(settings.DistanceFadeEnd, settings.DistanceFadeStart + 1.0f);
	}

	if (globals::state && globals::state->IsDeveloperMode() && ImGui::CollapsingHeader(T("feature.window_life.developer", "Developer"))) {
		ImGui::Spacing();
		ImGui::BeginDisabled(settings.AutomaticRoomSizing);
		ImGui::SliderFloat(T("feature.window_life.room_width", "Manual Room Width"), &settings.RoomWidth, 64.0f, 220.0f, "%.0f");
		ImGui::SliderFloat(T("feature.window_life.room_height", "Manual Room Height"), &settings.RoomHeight, 96.0f, 260.0f, "%.0f");
		ImGui::EndDisabled();
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("These controls apply only when Automatic Window Room Sizing is disabled. Automatic mode owns its stable calibration and cannot silently inherit stale manual values.");
		}
		ImGui::SliderFloat(T("feature.window_life.day_shadow", "Fallback Day Silhouette Darkness"), &settings.DayShadowStrength, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat(T("feature.window_life.night_shadow", "Fallback Night Silhouette Darkness"), &settings.NightShadowStrength, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("Used only if the authored occupant atlas is unavailable. Normal installations use Occupant Opacity above.");
		}
		ImGui::SliderFloat(T("feature.window_life.motion_speed", "Activity Speed"), &settings.MotionSpeed, 0.25f, 2.5f, "%.2f");
		ImGui::SliderFloat(T("feature.window_life.sun_glint", "Sun Glint Strength"), &settings.SunGlintStrength, 0.0f, 1.0f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat(T("feature.window_life.directional_reveal", "Directional Reveal Strength"), &settings.DirectionalRevealStrength, 0.0f, 0.60f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat(T("feature.window_life.room_variation", "Neighbouring Room Variation"), &settings.RoomVariationStrength, 0.0f, 0.35f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat(T("feature.window_life.layer_feather", "Close Cutout Feather"), &settings.CloseLayerFeather, 0.0f, 1.5f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		ImGui::SliderFloat("Multi-Window Mesh Threshold", &settings.MultiWindowMeshScale, 1.0f, 2.75f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("A mesh larger than this multiple of a normal window is treated as several windows. With a glow map or an installed glass mask, each window then gets its own fitted room instead of one room stretched across the whole mesh. Lower it if rooms look stretched or flat; 2.75 is the original behaviour.");
		}
		ImGui::Checkbox(T("feature.window_life.debug_detection", "Show Window Class Overlay"), &settings.DebugWindowDetection);
		if (auto _tt = Util::HoverTooltipWrapper()) {
			ImGui::TextWrapped("Blue/amber show glass/shallow tiers. Red means native layout rejected, cyan means native background only, and green means a full occupant-safe native layout. Procedural fallback remains independent of optional exact pane masks.");
		}
		if (settings.DebugWindowDetection) {
			const char* projectionViews[] = {
				"Class Overlay", "Base Room Coordinates", "Final Room Coordinates",
				"Framed Room UV", "Room Identity", "Room Basis", "Pane Mask",
				"Room Box Surface", "Room Box Depth", "Glow Guide Signal", "Multi-pane Group"
			};
			settings.DebugRoomProjection = std::clamp(settings.DebugRoomProjection, 0, 10);
			ImGui::Combo("Room Projection Debug", &settings.DebugRoomProjection,
				projectionViews, IM_ARRAYSIZE(projectionViews));
			if (auto _tt = Util::HoverTooltipWrapper()) {
				ImGui::TextWrapped("Developer diagnostic only. Use this to verify that every pane in a physical window shares the same room coordinates, identity and horizontal basis.");
			}
		}
		if (ImGui::Button(T("feature.window_life.clear_classifier", "Re-scan Window Materials"))) {
			classificationCache.clear();
			diag.classifyLines = 0;
			diag.geometryLines = 0;
			diag.loggedGeometry.clear();
			logger::info("[WindowLife] Material classification cache cleared.");
		}
	}
}

void WindowLife::LoadSettings(json& o_json)
{
	try {
		settings = o_json.is_object() ? o_json.get<Settings>() : Settings{};
	} catch (const nlohmann::json::exception& error) {
		logger::warn("[WindowLife] Malformed settings, retaining defaults: {}", error.what());
		settings = {};
	}
	const Settings defaults{};
	auto finiteRange = [](float value, float fallback, float low, float high) {
		return std::isfinite(value) ? std::clamp(value, low, high) : fallback;
	};
	settings.ParallaxDepth = finiteRange(settings.ParallaxDepth, defaults.ParallaxDepth, 0.0f, 240.0f);
	settings.InteriorParallaxDepth = finiteRange(settings.InteriorParallaxDepth, defaults.InteriorParallaxDepth, 0.0f, 240.0f);
	settings.Refraction = finiteRange(settings.Refraction, defaults.Refraction, 0.0f, 12.0f);
	settings.SilhouetteSoftness = finiteRange(settings.SilhouetteSoftness, defaults.SilhouetteSoftness, 0.005f, 0.25f);
	settings.DistanceFadeStart = finiteRange(settings.DistanceFadeStart, defaults.DistanceFadeStart, 0.0f, 32000.0f);
	settings.DistanceFadeEnd = finiteRange(settings.DistanceFadeEnd, std::max(defaults.DistanceFadeEnd, settings.DistanceFadeStart + 1.0f), settings.DistanceFadeStart + 1.0f, 64000.0f);
	settings.GlassStrength = finiteRange(settings.GlassStrength, defaults.GlassStrength, 0.0f, 1.0f);
	settings.GlassRoughness = finiteRange(settings.GlassRoughness, defaults.GlassRoughness, 0.04f, 1.0f);
	settings.GlassDistortion = finiteRange(settings.GlassDistortion, defaults.GlassDistortion, 0.0f, 0.15f);
	settings.RoomWidth = finiteRange(settings.RoomWidth, defaults.RoomWidth, 48.0f, 320.0f);
	settings.RoomHeight = finiteRange(settings.RoomHeight, defaults.RoomHeight, 72.0f, 360.0f);
	settings.PaneThreshold = finiteRange(settings.PaneThreshold, defaults.PaneThreshold, 0.0f, 0.55f);
	settings.PaneSoftness = finiteRange(settings.PaneSoftness, defaults.PaneSoftness, 0.03f, 0.50f);
	settings.OccupantOpacity = finiteRange(settings.OccupantOpacity, defaults.OccupantOpacity, 0.0f, 1.0f);
	settings.RoomVolumeStrength = finiteRange(settings.RoomVolumeStrength, defaults.RoomVolumeStrength, 0.0f, 1.0f);
	settings.WindowRecess = finiteRange(settings.WindowRecess, defaults.WindowRecess, 0.0f, 18.0f);
	settings.InteriorScale = finiteRange(settings.InteriorScale, defaults.InteriorScale, 1.0f, 2.50f);
	settings.InteriorSoftness = finiteRange(settings.InteriorSoftness, defaults.InteriorSoftness, 0.0f, 1.0f);
	settings.MotionSpeed = finiteRange(settings.MotionSpeed, defaults.MotionSpeed, 0.25f, 2.5f);
	settings.Refraction = finiteRange(settings.Refraction, defaults.Refraction, 0.0f, 12.0f);
	settings.DebugRoomProjection = std::clamp(settings.DebugRoomProjection, 0, 10);
	settings.MultiWindowMeshScale = finiteRange(settings.MultiWindowMeshScale, defaults.MultiWindowMeshScale, 1.0f, 2.75f);
	classificationCache.clear();
	activeDataValid = false;
	activeDataFrame = ~0u;
}

void WindowLife::SaveSettings(json& o_json)
{
	o_json = settings;
}

void WindowLife::RestoreDefaultSettings()
{
	settings = {};
	classificationCache.clear();
	activeDataValid = false;
	activeDataFrame = ~0u;
}

void WindowLife::SetupResources()
{
	if (!globals::d3d::device)
		return;

	// A renderer/device recreation must never retain views from the old D3D11
	// device. A failed recreation leaves WindowLife neutral rather than binding
	// a stale active SRV on a later Lighting draw.
	activeSRV = nullptr;
	neutralSRV = nullptr;
	activeBuffer = nullptr;
	neutralBuffer = nullptr;
	glassGrimeSRV = nullptr;
	occupantAtlasSRV = nullptr;
	curtainAtlasSRV = nullptr;
	roomAtlasSRV = nullptr;
	outdoorAtlasSRV = nullptr;
	outdoorNightAtlasSRV = nullptr;
	authoredMaskSRVs.clear();
	classificationCache.clear();
	frameBaseData = {};
	currentActiveData = {};
	activeDataValid = false;
	activeDataFrame = ~0u;
	uploadFailureLogged = false;
	invalidGeometryLogged = false;

	D3D11_BUFFER_DESC desc{};
	desc.ByteWidth = static_cast<UINT>(sizeof(PerGeometryData));
	desc.Usage = D3D11_USAGE_DYNAMIC;
	desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
	desc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
	desc.MiscFlags = D3D11_RESOURCE_MISC_BUFFER_STRUCTURED;
	desc.StructureByteStride = static_cast<UINT>(sizeof(PerGeometryData));

	const HRESULT activeResult = globals::d3d::device->CreateBuffer(&desc, nullptr, activeBuffer.put());
	const HRESULT neutralResult = globals::d3d::device->CreateBuffer(&desc, nullptr, neutralBuffer.put());
	if (FAILED(activeResult) || FAILED(neutralResult)) {
		logger::error("[WindowLife] Per-draw buffer creation failed (active 0x{:08X}, neutral 0x{:08X}); module renders neutral.",
			static_cast<std::uint32_t>(activeResult), static_cast<std::uint32_t>(neutralResult));
		activeBuffer = nullptr;
		neutralBuffer = nullptr;
		return;
	}

	D3D11_SHADER_RESOURCE_VIEW_DESC srvDesc{};
	srvDesc.Format = DXGI_FORMAT_UNKNOWN;
	srvDesc.ViewDimension = D3D11_SRV_DIMENSION_BUFFER;
	srvDesc.Buffer.FirstElement = 0;
	srvDesc.Buffer.NumElements = 1;
	const HRESULT activeViewResult = globals::d3d::device->CreateShaderResourceView(activeBuffer.get(), &srvDesc, activeSRV.put());
	const HRESULT neutralViewResult = globals::d3d::device->CreateShaderResourceView(neutralBuffer.get(), &srvDesc, neutralSRV.put());
	if (FAILED(activeViewResult) || FAILED(neutralViewResult)) {
		logger::error("[WindowLife] Per-draw SRV creation failed (active 0x{:08X}, neutral 0x{:08X}); module renders neutral.",
			static_cast<std::uint32_t>(activeViewResult), static_cast<std::uint32_t>(neutralViewResult));
		activeSRV = nullptr;
		neutralSRV = nullptr;
		return;
	}

	PerGeometryData neutral{};
	if (!UploadData(neutralBuffer.get(), neutral)) {
		activeSRV = nullptr;
		neutralSRV = nullptr;
		return;
	}
	frameBaseData = {};
	currentActiveData = {};
	activeDataValid = false;
	activeDataFrame = ~0u;

	// Optional Window Shadows-style masks are exact glass stencils only. They are
	// matched to the diffuse texture stem and bound on the real material draw;
	// they never promote a material into a window or control the room layout.
	authoredMaskSRVs.clear();
	const std::filesystem::path authoredMaskRoot = "Data\\Textures\\masks";
	std::size_t loadedMaskCount = 0u;
	std::size_t failedMaskCount = 0u;
	std::error_code maskDirectoryError;
	if (std::filesystem::exists(authoredMaskRoot, maskDirectoryError)) {
		for (std::filesystem::directory_iterator it(authoredMaskRoot, maskDirectoryError), end;
			 it != end && !maskDirectoryError;
			 it.increment(maskDirectoryError)) {
			std::error_code entryError;
			if (!it->is_regular_file(entryError) || entryError)
				continue;
			if (Lower(it->path().extension().string()) != ".dds")
				continue;

			const std::string maskKey = CanonicalWindowMaskKey(it->path().filename().string());
			if (maskKey.empty() || maskKey == "black")
				continue;

			DirectX::TexMetadata maskMetadata{};
			DirectX::ScratchImage maskImage;
			const HRESULT loadResult = DirectX::LoadFromDDSFile(
				it->path().c_str(),
				DirectX::DDS_FLAGS_NONE,
				&maskMetadata,
				maskImage);
			if (FAILED(loadResult) || maskMetadata.width < 8u || maskMetadata.height < 8u) {
				++failedMaskCount;
				logger::warn(
					"[WindowLife] Exact glass mask '{}' could not be loaded or is too small (HRESULT 0x{:08X}).",
					it->path().filename().string(),
					static_cast<std::uint32_t>(loadResult));
				continue;
			}

			// Label the mask's panes once (see BuildPaneBounds). Masks saved
			// without mips also get a chain here; the mask itself is kept at full
			// size for the exact pane clip.
			const std::size_t fileMips = maskMetadata.mipLevels;
			PaneBoundsResult paneBounds{};
			if (maskMetadata.dimension == DirectX::TEX_DIMENSION_TEXTURE2D && maskMetadata.arraySize == 1u) {
				DirectX::ScratchImage decompressed;
				DirectX::ScratchImage rgba;
				const DirectX::Image* top = maskImage.GetImage(0, 0, 0);
				if (top && DirectX::IsCompressed(top->format))
					top = SUCCEEDED(DirectX::Decompress(*top, DXGI_FORMAT_UNKNOWN, decompressed)) ? decompressed.GetImage(0, 0, 0) : nullptr;
				if (top && top->format != DXGI_FORMAT_R8G8B8A8_UNORM)
					top = SUCCEEDED(DirectX::Convert(*top, DXGI_FORMAT_R8G8B8A8_UNORM, DirectX::TEX_FILTER_DEFAULT, DirectX::TEX_THRESHOLD_DEFAULT, rgba)) ? rgba.GetImage(0, 0, 0) : nullptr;
				if (top && top->pixels) {
					paneBounds = BuildPaneBounds(top->pixels, top->rowPitch, top->width, top->height);
					if (fileMips <= 1u) {
						DirectX::ScratchImage maskMipChain;
						if (SUCCEEDED(DirectX::GenerateMipMaps(*top, DirectX::TEX_FILTER_DEFAULT, 0u, maskMipChain)) && maskMipChain.GetImageCount() > 1u) {
							maskImage = std::move(maskMipChain);
							maskMetadata = maskImage.GetMetadata();
						}
					}
				}
			}

			winrt::com_ptr<ID3D11ShaderResourceView> boundsSRV;
			if (paneBounds.panes > 0 && !paneBounds.texels.empty()) {
				D3D11_TEXTURE2D_DESC boundsDesc{};
				boundsDesc.Width = static_cast<UINT>(paneBounds.width);
				boundsDesc.Height = static_cast<UINT>(paneBounds.height);
				boundsDesc.MipLevels = 1;
				boundsDesc.ArraySize = 1;
				boundsDesc.Format = DXGI_FORMAT_R16G16B16A16_UNORM;
				boundsDesc.SampleDesc.Count = 1;
				boundsDesc.Usage = D3D11_USAGE_IMMUTABLE;
				boundsDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
				D3D11_SUBRESOURCE_DATA boundsData{};
				boundsData.pSysMem = paneBounds.texels.data();
				boundsData.SysMemPitch = static_cast<UINT>(paneBounds.width * 4u * sizeof(std::uint16_t));
				winrt::com_ptr<ID3D11Texture2D> boundsTexture;
				if (SUCCEEDED(globals::d3d::device->CreateTexture2D(&boundsDesc, &boundsData, boundsTexture.put())) &&
					FAILED(globals::d3d::device->CreateShaderResourceView(boundsTexture.get(), nullptr, boundsSRV.put())))
					boundsSRV = nullptr;
			}

			winrt::com_ptr<ID3D11ShaderResourceView> maskSRV;
			const HRESULT srvResult = DirectX::CreateShaderResourceView(
				globals::d3d::device,
				maskImage.GetImages(),
				maskImage.GetImageCount(),
				maskMetadata,
				maskSRV.put());
			if (FAILED(srvResult)) {
				++failedMaskCount;
				logger::warn(
					"[WindowLife] Exact glass mask '{}' SRV creation failed (HRESULT 0x{:08X}).",
					it->path().filename().string(),
					static_cast<std::uint32_t>(srvResult));
				continue;
			}

			logger::info(
				"[WindowLife] Glass mask '{}' matches diffuse textures named '{}' ({}x{}, {} mips in file, {} in use); {} panes found{}",
				it->path().filename().string(), maskKey, maskMetadata.width, maskMetadata.height, fileMips, maskMetadata.mipLevels,
				paneBounds.panes, boundsSRV ? "" : " (pane lookup unavailable, using the edge search)");
			authoredMaskSRVs.insert_or_assign(maskKey, AuthoredMask{ std::move(maskSRV), std::move(boundsSRV) });
			++loadedMaskCount;
		}
	}
	if (maskDirectoryError) {
		logger::warn(
			"[WindowLife] Exact glass-mask directory scan failed (error {}).",
			maskDirectoryError.value());
	}
	logger::info(
		"[WindowLife] Loaded {} optional exact glass masks for final pane clipping ({} failed); procedural room placement remains authoritative.",
		loadedMaskCount,
		failedMaskCount);
	classificationCache.clear();

	// Prefer precompressed DDS atlases with authored mip chains. This avoids WIC
	// decode plus runtime RGBA mip generation and keeps WindowLife's largest
	// optional assets compressed in GPU memory. PNG remains a non-fatal fallback
	// for older packages and for replacement authors who have not converted yet.
	const auto loadAtlas = [&](const std::filesystem::path& ddsPath,
							   const std::filesystem::path& pngFallbackPath,
							   std::string_view label,
							   winrt::com_ptr<ID3D11ShaderResourceView>& target,
							   bool analyticFallback) {
		target = nullptr;

		if (!ddsPath.empty()) {
			DirectX::ScratchImage ddsImage;
			DirectX::TexMetadata ddsMetadata{};
			const HRESULT ddsLoadResult = DirectX::LoadFromDDSFile(
				ddsPath.c_str(),
				DirectX::DDS_FLAGS_NONE,
				&ddsMetadata,
				ddsImage);
			if (SUCCEEDED(ddsLoadResult) &&
				ddsMetadata.width == 2048u &&
				ddsMetadata.height == 2048u &&
				ddsMetadata.dimension == DirectX::TEX_DIMENSION_TEXTURE2D &&
				ddsMetadata.arraySize == 1u &&
				ddsMetadata.mipLevels > 1u &&
				DirectX::IsCompressed(ddsMetadata.format)) {
				const DXGI_FORMAT srgbFormat = DirectX::MakeSRGB(ddsMetadata.format);
				if (srgbFormat != DXGI_FORMAT_UNKNOWN && ddsImage.OverrideFormat(srgbFormat)) {
					ddsMetadata = ddsImage.GetMetadata();
					const HRESULT srvResult = DirectX::CreateShaderResourceView(
						globals::d3d::device,
						ddsImage.GetImages(),
						ddsImage.GetImageCount(),
						ddsMetadata,
						target.put());
					if (SUCCEEDED(srvResult) && target) {
						logger::info(
							"[WindowLife] {} DDS atlas '{}' uploaded as format {} with {} authored mip levels.",
							label,
							ddsPath.string(),
							static_cast<std::uint32_t>(ddsMetadata.format),
							ddsMetadata.mipLevels);
						return;
					}
					logger::warn(
						"[WindowLife] {} DDS atlas SRV creation failed (HRESULT 0x{:08X}); trying PNG fallback.",
						label,
						static_cast<std::uint32_t>(srvResult));
					target = nullptr;
				} else {
					logger::warn(
						"[WindowLife] {} DDS atlas '{}' could not be exposed as sRGB; trying PNG fallback.",
						label,
						ddsPath.string());
				}
			} else if (SUCCEEDED(ddsLoadResult)) {
				logger::warn(
					"[WindowLife] {} DDS atlas '{}' must be one compressed 2048x2048 2D texture with authored mips (found {}x{}, array {}, mips {}, format {}); trying PNG fallback.",
					label,
					ddsPath.string(),
					ddsMetadata.width,
					ddsMetadata.height,
					ddsMetadata.arraySize,
					ddsMetadata.mipLevels,
					static_cast<std::uint32_t>(ddsMetadata.format));
			} else {
				logger::info(
					"[WindowLife] Optional {} DDS atlas '{}' was not loaded (HRESULT 0x{:08X}); trying PNG compatibility fallback.",
					label,
					ddsPath.string(),
					static_cast<std::uint32_t>(ddsLoadResult));
			}
		}

		DirectX::TexMetadata pngMetadata{};
		DirectX::ScratchImage pngImage;
		DirectX::ScratchImage pngMipChain;
		const HRESULT pngLoadResult = DirectX::LoadFromWICFile(
			pngFallbackPath.c_str(),
			DirectX::WIC_FLAGS_FORCE_SRGB,
			&pngMetadata,
			pngImage);
		if (FAILED(pngLoadResult) || pngMetadata.width == 0u || pngMetadata.width != pngMetadata.height) {
			logger::warn(
				"[WindowLife] {} PNG fallback '{}' unavailable or not square (HRESULT 0x{:08X}, {}x{}); {} fallback remains active.",
				label,
				pngFallbackPath.string(),
				static_cast<std::uint32_t>(pngLoadResult),
				pngMetadata.width,
				pngMetadata.height,
				analyticFallback ? "analytic" : "procedural");
			return;
		}

		// The shaders use a 2048 atlas/tile padding contract. Normalize valid
		// square replacement artwork once at load time before generating mips.
		if (pngMetadata.width != 2048u) {
			DirectX::ScratchImage resized;
			const HRESULT resizeResult = DirectX::Resize(
				*pngImage.GetImage(0, 0, 0), 2048u, 2048u,
				static_cast<DirectX::TEX_FILTER_FLAGS>(DirectX::TEX_FILTER_CUBIC | DirectX::TEX_FILTER_SEPARATE_ALPHA), resized);
			if (FAILED(resizeResult)) {
				logger::warn("[WindowLife] {} atlas resize failed (HRESULT 0x{:08X}); fallback remains active.",
					label, static_cast<std::uint32_t>(resizeResult));
				return;
			}
			logger::info("[WindowLife] Normalized {} atlas from {}x{} to 2048x2048", label, pngMetadata.width, pngMetadata.height);
			pngImage = std::move(resized);
			pngMetadata = pngImage.GetMetadata();
		}

		const HRESULT mipResult = DirectX::GenerateMipMaps(
			pngImage.GetImages(),
			pngImage.GetImageCount(),
			pngMetadata,
			static_cast<DirectX::TEX_FILTER_FLAGS>(
				DirectX::TEX_FILTER_CUBIC | DirectX::TEX_FILTER_SEPARATE_ALPHA),
			0u,
			pngMipChain);
		const bool hasMips = SUCCEEDED(mipResult) && pngMipChain.GetImageCount() > 1u;
		const DirectX::Image* uploadImages = hasMips ? pngMipChain.GetImages() : pngImage.GetImages();
		const std::size_t uploadCount = hasMips ? pngMipChain.GetImageCount() : pngImage.GetImageCount();
		const DirectX::TexMetadata& uploadMetadata = hasMips ? pngMipChain.GetMetadata() : pngMetadata;
		const HRESULT srvResult = DirectX::CreateShaderResourceView(
			globals::d3d::device,
			uploadImages,
			uploadCount,
			uploadMetadata,
			target.put());
		if (FAILED(srvResult)) {
			logger::warn(
				"[WindowLife] {} PNG fallback SRV creation failed (HRESULT 0x{:08X}); {} fallback remains active.",
				label,
				static_cast<std::uint32_t>(srvResult),
				analyticFallback ? "analytic" : "procedural");
			target = nullptr;
			return;
		}
		logger::info(
			"[WindowLife] {} PNG compatibility atlas uploaded with {} mip levels.",
			label,
			uploadMetadata.mipLevels);
	};

	loadAtlas(
		"Data\\Shaders\\WindowLife\\RoomAtlas_2k.dds",
		"Data\\Shaders\\WindowLife\\RoomAtlas.png",
		"Room",
		roomAtlasSRV,
		false);
	loadAtlas(
		"Data\\Shaders\\WindowLife\\OutdoorAtlas_2k.dds",
		"Data\\Shaders\\WindowLife\\OutdoorAtlas.png",
		"Outdoor view",
		outdoorAtlasSRV,
		false);
	loadAtlas(
		"Data\\Shaders\\WindowLife\\OutdoorAtlasNight_2k.dds",
		"Data\\Shaders\\WindowLife\\OutdoorAtlasNight.png",
		"Outdoor night view",
		outdoorNightAtlasSRV,
		false);
	loadAtlas(
		{},
		"Data\\Shaders\\WindowLife\\OccupantAtlas.png",
		"Occupant",
		occupantAtlasSRV,
		true);
	loadAtlas(
		"Data\\Shaders\\WindowLife\\CurtainAtlas_high-fidelity-2k.dds",
		"Data\\Shaders\\WindowLife\\CurtainAtlas.png",
		"Curtain",
		curtainAtlasSRV,
		true);

	// Small neutral mask texture replacing per-pixel high-frequency analytic
	// grime. Generate a filtered mip chain at load time; failure is non-fatal and
	// the shader retains its restrained low-frequency analytic fallback.
	glassGrimeSRV = nullptr;
	{
		const std::filesystem::path grimePath = "Data\\Shaders\\WindowLife\\GlassGrime_1k.png";
		DirectX::TexMetadata grimeMetadata{};
		DirectX::ScratchImage grimeImage;
		DirectX::ScratchImage grimeMipChain;
		const HRESULT grimeLoadResult = DirectX::LoadFromWICFile(
			grimePath.c_str(),
			DirectX::WIC_FLAGS_FORCE_RGB,
			&grimeMetadata,
			grimeImage);
		if (SUCCEEDED(grimeLoadResult) && grimeMetadata.width == 1024u && grimeMetadata.height == 1024u) {
			const HRESULT mipResult = DirectX::GenerateMipMaps(
				grimeImage.GetImages(),
				grimeImage.GetImageCount(),
				grimeMetadata,
				DirectX::TEX_FILTER_CUBIC,
				0u,
				grimeMipChain);
			const bool hasMips = SUCCEEDED(mipResult) && grimeMipChain.GetImageCount() > 1u;
			const auto* images = hasMips ? grimeMipChain.GetImages() : grimeImage.GetImages();
			const std::size_t imageCount = hasMips ? grimeMipChain.GetImageCount() : grimeImage.GetImageCount();
			const auto& metadata = hasMips ? grimeMipChain.GetMetadata() : grimeMetadata;
			const HRESULT srvResult = DirectX::CreateShaderResourceView(
				globals::d3d::device,
				images,
				imageCount,
				metadata,
				glassGrimeSRV.put());
			if (FAILED(srvResult))
				glassGrimeSRV = nullptr;
		}
		logger::info(
			"[WindowLife] Mip-filtered glass grime texture '{}' {}.",
			grimePath.string(),
			glassGrimeSRV ? "ready" : "unavailable; analytic fallback active");
	}

	logger::info(
		"[WindowLife] Layered-window GPU resources ready (PS t{} grime={}, t{} occupants={}, t{} curtains={}, t{} optional exact pane clip, t{} room atlas={}, t{} structured SRV, 256-byte per-draw payload; FeatureData b6 unchanged).",
		kGlassGrimeSRVSlot,
		glassGrimeSRV ? "ready" : "analytic",
		kOccupantAtlasSRVSlot,
		occupantAtlasSRV ? "ready" : "analytic",
		kCurtainAtlasSRVSlot,
		curtainAtlasSRV ? "ready" : "analytic",
		kAuthoredMaskSRVSlot,
		kRoomAtlasSRVSlot,
		roomAtlasSRV ? "ready" : "fallback",
		kPerDrawSRVSlot);
}

float WindowLife::GetDayNightBlend(float hour)
{
	if (!std::isfinite(hour))
		hour = 12.0f;
	// Calendar continues indoors; the active climate supplies twilight timing
	// for both the outdoor-view atlas and exterior room lighting.
	if (const auto* sky = globals::game::sky; sky && sky->currentClimate) {
		const auto& timing = sky->currentClimate->timing;
		const float daylight = DaylightScale(
			hour, timing.sunrise.begin / 6.0f, timing.sunrise.end / 6.0f,
			timing.sunset.begin / 6.0f, timing.sunset.end / 6.0f);
		return std::clamp((1.0f - daylight) / 0.95f, 0.0f, 1.0f);
	}
	hour = std::fmod(std::max(hour, 0.0f), 24.0f);
	if (hour < 5.0f)
		return 1.0f;
	if (hour < 8.0f)
		return 1.0f - SmoothRange(5.0f, 8.0f, hour);
	if (hour < 17.0f)
		return 0.0f;
	if (hour < 20.0f)
		return SmoothRange(17.0f, 20.0f, hour);
	return 1.0f;
}

float WindowLife::GetActivityForHour(float hour, const Settings& s)
{
	hour = std::fmod(std::max(hour, 0.0f), 24.0f);
	if (hour < 5.0f)
		return s.LateNightActivity;
	if (hour < 7.0f) {
		const float t = SmoothRange(5.0f, 7.0f, hour);
		return std::lerp(s.LateNightActivity, s.DayActivity, t);
	}
	if (hour < 17.0f)
		return s.DayActivity;
	if (hour < 20.0f) {
		const float t = SmoothRange(17.0f, 20.0f, hour);
		return std::lerp(s.DayActivity, s.EveningActivity, t);
	}
	if (hour < 23.0f)
		return s.EveningActivity;
	const float t = SmoothRange(23.0f, 24.0f, hour);
	return std::lerp(s.EveningActivity, s.LateNightActivity, t);
}

void WindowLife::RefreshFrameBaseData()
{
	if (!activeBuffer)
		return;

	const std::uint32_t frame = globals::state ? globals::state->frameCount : 0u;
	if (activeDataFrame == frame)
		return;

	float hour = 12.0f;
	auto* calendar = globals::game::calendar;
	if (!calendar)
		calendar = RE::Calendar::GetSingleton();
	if (calendar)
		hour = calendar->GetHour();
	if (!std::isfinite(hour))
		hour = 12.0f;

	const float night = GetDayNightBlend(hour);
	const float shadowStrength = std::lerp(
		std::clamp(settings.DayShadowStrength, 0.0f, 1.0f),
		std::clamp(settings.NightShadowStrength, 0.0f, 1.0f),
		night);
	const float activity = std::clamp(GetActivityForHour(hour, settings), 0.0f, 1.0f);

	frameBaseData = {};
	// Match SharedData::InInterior exactly. Select the atlas once per frame and
	// reuse the room-atlas slot: no additional resource slots or per-draw ABI growth.
	const bool interiorView = Util::IsInterior();
	frameBaseData.Runtime0 = {
		settings.EnableWindowLife ? 1.0f : 0.0f,
		0.6180339887f,
		shadowStrength,
		activity
	};
	frameBaseData.Optics0 = {
		std::clamp(interiorView ? settings.InteriorParallaxDepth : settings.ParallaxDepth, 0.0f, 240.0f),
		std::clamp(settings.SilhouetteSoftness, 0.005f, 0.25f),
		std::clamp(settings.Refraction, 0.0f, 12.0f),
		std::clamp(settings.HumanScale, 0.5f, 1.6f)
	};
	frameBaseData.Surface0 = {
		std::max(settings.DistanceFadeStart, 0.0f),
		std::max(settings.DistanceFadeEnd, settings.DistanceFadeStart + 1.0f),
		std::clamp(settings.PaneThreshold, 0.0f, 1.0f),
		std::clamp(settings.PaneSoftness, 0.01f, 1.0f)
	};
	// Runtime1.x is a debug mode, not a rendering parameter. Existing shaders
	// treat any non-zero value as the historical class overlay, so mode 1 keeps
	// that behaviour and modes 2..10 expose room/material diagnostics.
	const int debugMode = settings.DebugWindowDetection ? std::clamp(settings.DebugRoomProjection, 0, 10) + 1 : 0;
	frameBaseData.Runtime1 = {
		static_cast<float>(debugMode),
		std::clamp(settings.RoomWidth, 48.0f, 320.0f),
		std::clamp(settings.RoomHeight, 72.0f, 360.0f),
		std::clamp(settings.MotionSpeed, 0.1f, 4.0f)
	};
	frameBaseData.Glass0 = {
		std::clamp(settings.GlassStrength, 0.0f, 1.0f),
		std::clamp(settings.GlassReflectionBoost, 0.0f, 2.0f),
		std::clamp(settings.GlassRoughness, 0.04f, 1.0f),
		std::clamp(settings.GlassTransmission, 0.5f, 1.0f)
	};
	frameBaseData.Glass1 = {
		std::clamp(settings.GlassDirtStrength, 0.0f, 1.0f),
		std::clamp(settings.GlassDistortion, 0.0f, 0.15f),
		std::clamp(settings.GlassNormalRetention, 0.0f, 1.0f),
		settings.SuppressWindowAutoPOM ? 1.0f : 0.0f
	};
	frameBaseData.Eligibility0 = {
		std::max(settings.MinShallowWindowRadius, 0.0f),
		std::max(settings.MinFullWindowRadius, settings.MinShallowWindowRadius + 1.0f),
		std::clamp(settings.FullWindowVerticality, 0.0f, 1.0f),
		settings.EnableArchitecturalGlass ? 1.0f : 0.0f
	};
	frameBaseData.Interior0 = {
		std::clamp(settings.CurtainStrength, 0.0f, 1.0f),
		std::clamp(settings.RoomDepthStrength, 0.0f, 0.60f),
		settings.AutomaticRoomSizing ? 1.0f : 0.0f,
		settings.EnableInteriorPassers ? 1.0f : 0.0f
	};
	frameBaseData.Asset0 = {
		0.0f,
		(interiorView ? settings.EnableOutdoorViews && outdoorAtlasSRV : settings.EnableAuthoredRooms && roomAtlasSRV) ? 1.0f : 0.0f,
		std::clamp(interiorView ? settings.OutdoorViewStrength : settings.AuthoredRoomStrength, 0.0f, 2.0f),
		0.0f
	};
	frameBaseData.Presentation0 = {
		std::clamp(settings.InteriorContrast, 0.50f, 2.0f),
		interiorView ? std::clamp(settings.OutdoorViewEmission, 0.0f, 8.0f) : std::clamp(settings.InteriorEmission, 0.0f, 6.0f),
		std::clamp(settings.OccupantOpacity, 0.0f, 1.0f),
		std::clamp(settings.InteriorScale, 1.0f, 2.5f)
	};
	const bool snowing = GetSnowWeatherAmount() > 0.01f;
	frameBaseData.Layout0 = {
		0.0f,
		0.0f,
		night,
		snowing ? 1.0f : 0.0f
	};
	frameBaseData.Fidelity0 = {
		std::clamp(settings.EnvironmentReflectionStrength, 0.0f, 1.5f),
		std::clamp(settings.InteriorLightingResponse, 0.0f, 1.0f),
		std::clamp(settings.WeatherGlassResponse, 0.0f, 1.0f),
		std::clamp(settings.SunGlintStrength, 0.0f, 1.0f)
	};
	frameBaseData.Fidelity1 = {
		std::clamp(settings.DirectionalRevealStrength, 0.0f, 0.60f),
		std::clamp(settings.RoomVariationStrength, 0.0f, 0.35f),
		std::clamp(settings.CloseLayerFeather, 0.0f, 1.5f),
		interiorView ? 1.0f : 0.0f
	};
	frameBaseData.Presentation1 = {
		std::clamp(settings.InteriorSoftness, 0.0f, 1.0f),
		std::clamp(settings.RoomVolumeStrength, 0.0f, 1.0f),
		std::clamp(settings.WindowRecess, 0.0f, 18.0f),
		0.0f
	};

	activeDataFrame = frame;
}

bool WindowLife::UploadData(ID3D11Buffer* buffer, const PerGeometryData& data) const
{
	if (!buffer || !globals::d3d::context)
		return false;

	D3D11_MAPPED_SUBRESOURCE mapped{};
	const HRESULT mapResult = globals::d3d::context->Map(buffer, 0u, D3D11_MAP_WRITE_DISCARD, 0u, &mapped);
	if (FAILED(mapResult)) {
		if (!uploadFailureLogged) {
			logger::error("[WindowLife] Per-draw buffer upload failed (HRESULT 0x{:08X}); binding neutral until resources recover.",
				static_cast<std::uint32_t>(mapResult));
			uploadFailureLogged = true;
		}
		return false;
	}
	std::memcpy(mapped.pData, &data, sizeof(data));
	globals::d3d::context->Unmap(buffer, 0u);
	uploadFailureLogged = false;
	return true;
}

void WindowLife::BindNeutral() const
{
	if (!globals::d3d::context)
		return;
	ID3D11ShaderResourceView* srvs[7] = {
		glassGrimeSRV.get(),
		occupantAtlasSRV.get(),
		curtainAtlasSRV.get(),
		nullptr,
		nullptr,
		neutralSRV.get(),
		nullptr
	};
	// Two runs: t46-t49 (grime, occupants, curtains, exact mask) and t56-t58 (room atlas, per-draw data, pane bounds).
	globals::d3d::context->PSSetShaderResources(kGlassGrimeSRVSlot, 4, srvs);
	globals::d3d::context->PSSetShaderResources(kRoomAtlasSRVSlot, 3, srvs + 4);
}

const WindowLife::AuthoredMask* WindowLife::GetAuthoredMask(const Classification& classification) const
{
	// Asset0.z is authored room visibility, NOT a mask-eligibility flag.
	// The previous guard disabled the user's exact glass masks at the default
	// room visibility of 1.0. Masks clip glass only; they never drive room art.
	if (!settings.UseExactGlassMasks || !classification.hasAuthoredMask || classification.authoredMaskKey.empty())
		return nullptr;
	const auto it = authoredMaskSRVs.find(classification.authoredMaskKey);
	return it != authoredMaskSRVs.end() ? &it->second : nullptr;
}

void WindowLife::BindActive(const Classification& classification) const
{
	if (!globals::d3d::context)
		return;
	if (!activeSRV) {
		BindNeutral();
		return;
	}
	const AuthoredMask* authoredMask = GetAuthoredMask(classification);
	ID3D11ShaderResourceView* srvs[7] = {
		glassGrimeSRV.get(),
		occupantAtlasSRV.get(),
		frameBaseData.Fidelity1.w > 0.5f ? outdoorNightAtlasSRV.get() : curtainAtlasSRV.get(),
		authoredMask ? authoredMask->mask.get() : nullptr,
		frameBaseData.Asset0.y > 0.5f ? (frameBaseData.Fidelity1.w > 0.5f ? outdoorAtlasSRV.get() : roomAtlasSRV.get()) : nullptr,
		activeSRV.get(),
		authoredMask ? authoredMask->paneBounds.get() : nullptr
	};
	// Two runs: t46-t49 (grime, occupants, curtains, exact mask) and t56-t58 (room atlas, per-draw data, pane bounds).
	globals::d3d::context->PSSetShaderResources(kGlassGrimeSRVSlot, 4, srvs);
	globals::d3d::context->PSSetShaderResources(kRoomAtlasSRVSlot, 3, srvs + 4);
}

WindowLife::Classification WindowLife::ClassifyMaterial(const RE::BSLightingShaderMaterialBase* material)
{
	Classification result{};
	if (!material)
		return result;
	++diag.newMaterials;

	auto* textures = material->textureSet.get();
	if (!textures) {
		++diag.noTextureSet;
		return result;
	}

	std::string allPaths;
	std::string diffusePath;
	std::string glowPath;

	const auto appendPath = [&](RE::BSTextureSet::Texture slot, std::string* capture = nullptr) {
		const char* raw = textures->GetTexturePath(slot);
		if (!raw || !*raw)
			return;
		std::string p = Lower(raw);
		if (capture)
			*capture = p;
		if (!allPaths.empty())
			allPaths.push_back('|');
		allPaths += p;
	};

	appendPath(RE::BSTextureSet::Texture::kDiffuse, &diffusePath);
	appendPath(RE::BSTextureSet::Texture::kGlowMap, &glowPath);
	appendPath(RE::BSTextureSet::Texture::kNormal);
	appendPath(RE::BSTextureSet::Texture::kEnvironmentMask);
	appendPath(RE::BSTextureSet::Texture::kEnvironment);

	if (allPaths.empty()) {
		++diag.noTexturePaths;
		return result;
	}

	// The diffuse material identifies the surface being drawn. A glow/normal slot
	// belonging to another atlas is not enough to turn roofs, stucco, doors,
	// shutters, trim, floors or whole facades into windows.
	const bool strongWindow = ContainsAny(diffusePath, { "window", "windows", "windowpane", "window_pane", "glasspane", "glass_pane", "glazing" });
	const bool glass = ContainsAny(allPaths, { "glass", "stainedglass", "stained_glass" });
	const bool architecture = ContainsAny(allPaths, { "architecture", "architectural", "farmhouse", "whiterun", "solitude", "windhelm", "riften", "markarth",
														"winterhold", "college", "castle", "house", "houses", "inn", "shop", "temple", "fort",
														"building", "buildings", "village", "town", "city", "exterior", "facade", "dwemer", "nordic" });
	const bool diffuseArchitecture = ContainsAny(diffusePath, { "architecture", "architectural", "farmhouse", "whiterun", "solitude", "windhelm", "riften", "markarth",
																  "winterhold", "college", "castle", "house", "houses", "inn", "shop", "temple", "fort",
																  "building", "buildings", "village", "town", "city", "exterior", "facade", "dwemer", "nordic" });
	const bool obviousNonBuildingGlass = ContainsAny(allPaths, { "bottle", "potion", "alchemy", "clutter", "weapon", "armor", "shield", "crystal", "gem",
																   "ice", "eye", "water", "magic", "spell", "effect", "decal", "lantern", "chandelier", "candle", "torch" });
	// Window-shadow/interior mods commonly add separate proxy geometry whose
	// diffuse texture is a binary pane mask. The proxy filename still contains
	// "window", but applying WindowLife to that draw exposes the helper mesh in
	// front of or through the building. It is not the authored glass surface.
	const bool windowProxyMask = ContainsAny(diffusePath, { "\\masks\\", "/masks/", "_mask.dds", "windowshadow", "window_shadow" });
	// Closed/boarded shutters remain opaque architecture even when a texture
	// replacer preserves "window" in the filename.
	const bool closedWindowSurface = ContainsAny(diffusePath, { "shutter", "closedwindow", "closed_window", "windowclosed",
																  "window_closed", "boarded", "windowboard", "window_board" });
	const bool hasGlowTexture = !glowPath.empty();
	// Some Skyrim building materials name the diffuse after the facade but put
	// the actual pane evidence in a dedicated window glow map. The glow guide is
	// accepted only with an architectural diffuse and remains clipped per pixel
	// by PaneMask; a generic building path or unrelated glow map is insufficient.
	const bool windowGlowGuide = hasGlowTexture && ContainsAny(glowPath, { "window", "windows", "windowpane", "window_pane", "glasspane", "glass_pane", "glazing" });
	std::string authoredMaskKey = CanonicalWindowMaskKey(diffusePath);
	// Texture replacers often append a short letter variant to the vanilla stem
	// (riftenwindows02 -> riftenwindows02ay) while the installed mask keeps the
	// vanilla name. Fall back to the stem when the exact name has no mask.
	if (!authoredMaskKey.empty() && !authoredMaskSRVs.contains(authoredMaskKey)) {
		std::size_t end = authoredMaskKey.size();
		std::size_t letters = 0;
		while (end > 0 && letters < 3 && std::isalpha(static_cast<unsigned char>(authoredMaskKey[end - 1]))) {
			--end;
			++letters;
		}
		if (letters > 0 && end > 0 && std::isdigit(static_cast<unsigned char>(authoredMaskKey[end - 1]))) {
			std::string stem = authoredMaskKey.substr(0, end);
			if (authoredMaskSRVs.contains(stem)) {
				if (settings.DebugWindowDetection)
					logger::info("[WindowLife] '{}' uses the '{}' glass mask (variant name)", authoredMaskKey, stem);
				authoredMaskKey = std::move(stem);
			}
		}
	}
	const bool hasAuthoredMask =
		settings.UseExactGlassMasks &&
		!authoredMaskKey.empty() &&
		authoredMaskSRVs.contains(authoredMaskKey);

	// Texture replacers commonly rename the diffuse while retaining a paired
	// window mask. Treat that exact diffuse/mask pairing as strong material
	// evidence; a generic glow slot alone remains deliberately insufficient.
	const bool authoredWindowMaterial = hasAuthoredMask && !closedWindowSurface;

	if (strongWindow)
		result.score += 8;
	if (glass)
		result.score += 2;
	if (architecture)
		result.score += 3;
	if (hasGlowTexture)
		result.score += 2;
	if (authoredWindowMaterial)
		result.score += 8;
	if (architecture && hasGlowTexture)
		result.score += 3;
	if (obviousNonBuildingGlass && !strongWindow)
		result.score -= 8;
	if (windowProxyMask || closedWindowSurface)
		result.score -= 16;

	const bool architecturalGlass = architecture && glass && !obviousNonBuildingGlass;
	// Architecture is context, never proof. Treating the broad architecture path
	// as sufficient classified doors, roofs, gravestones and entire facade draws
	// as windows. Require a window/glass token on the diffuse or paired glow,
	// or a dedicated authored pane mask matching this material.
	result.isWindow = !windowProxyMask && !closedWindowSurface &&
	                  (strongWindow || architecturalGlass || authoredWindowMaterial ||
						  (diffuseArchitecture && windowGlowGuide && !obviousNonBuildingGlass));
	result.hasGlowTexture = hasGlowTexture;
	// A diffuse-matched exact mask can admit a texture replacer's window into
	// the shallow room tier. Closed/helper proxy surfaces remain excluded above;
	// the mask clips final glass pixels and never moves the room aperture.
	result.hasAuthoredMask = result.isWindow && hasAuthoredMask;
	result.authoredMaskKey = result.hasAuthoredMask ? authoredMaskKey : std::string{};
	result.explicitWindow = strongWindow || (diffuseArchitecture && windowGlowGuide && !obviousNonBuildingGlass);
	result.diffuseNamedWindow = strongWindow;
	result.namedGlass = glass;
	result.diffuseNamedGlass = ContainsAny(diffusePath, { "glass", "stainedglass", "stained_glass" });
	// Cache the material side of the room identity once. The geometry instance
	// contributes its own stable salt in UpdateAndBindActive; separating the two
	// prevents per-pixel texture evidence from changing room identity.
	result.materialIdentity = material->hashKey != 0 ? material->hashKey : StableHash32(!diffusePath.empty() ? diffusePath : allPaths);
	if (ContainsAny(allPaths, { "markarth", "dwemer" }))
		result.roomFamily = 4;
	else if (ContainsAny(allPaths, { "solitude", "castle", "imperial", "palace", "noble" }))
		result.roomFamily = 1;
	else if (ContainsAny(allPaths, { "riften", "canal" }))
		result.roomFamily = 2;
	else if (ContainsAny(allPaths, { "windhelm", "snowquarter" }))
		result.roomFamily = 3;
	else if (ContainsAny(allPaths, { "inn", "shop", "tavern", "merchant", "alchemy" }))
		result.roomFamily = 5;

	// Material tier is only a ceiling. Geometry radius and surface orientation can
	// downgrade it later; they can never promote decorative/non-window materials.
	if (strongWindow)
		result.materialTier = 3;
	else if (architecturalGlass || (diffuseArchitecture && windowGlowGuide) || authoredWindowMaterial)
		// Exact diffuse-matched masks admit a room but do not infer a safe
		// occupant layer across a large facade draw.
		result.materialTier = 2;

	result.evidence = !diffusePath.empty() ? diffusePath : (!glowPath.empty() ? glowPath : allPaths);

	if (settings.DebugWindowDetection && result.isWindow && diag.classifyLines < kMaxClassifyLogLines) {
		++diag.classifyLines;
		logger::info(
			"[WindowLife] CLASSIFY {} tier={} score={} hash={:08X} arch={} glass={} glow={} mask={} explicit={} proxy={} closed={} texture='{}' layout='{}' glowmap='{}'",
			result.isWindow ? "WINDOW" : "skip",
			result.materialTier,
			result.score,
			material->hashKey,
			architecture ? 1 : 0,
			glass ? 1 : 0,
			hasGlowTexture ? 1 : 0,
			result.hasAuthoredMask ? 1 : 0,
			result.explicitWindow ? 1 : 0,
			windowProxyMask ? 1 : 0,
			closedWindowSurface ? 1 : 0,
			result.evidence,
			result.hasAuthoredMask ? "procedural + exact clip" : "procedural",
			glowPath);
	}

	return result;
}

const WindowLife::Classification& WindowLife::GetClassification(const RE::BSLightingShaderMaterialBase* material)
{
	static const Classification kNone{};
	if (!material)
		return kNone;

	// hashKey is not a unique material-object identity across replacers and
	// derived lighting materials. Sharing classification by hash can promote
	// a neighbouring facade/mullion material into the window shader.
	const std::uintptr_t key = reinterpret_cast<std::uintptr_t>(material);

	auto it = classificationCache.find(key);
	if (it != classificationCache.end())
		return it->second;

	auto emplaceResult = classificationCache.emplace(key, ClassifyMaterial(material));
	return emplaceResult.first->second;
}

void WindowLife::UpdateAndBindActive(const Classification& classification, const RE::BSGeometry* geometry)
{
	RefreshFrameBaseData();

	PerGeometryData data = frameBaseData;
	const auto sharedWindow = ResolveSharedWindowBounds(geometry, settings.GroupMultiPaneWindows);
	const float geometryRadius = sharedWindow.radius;

	int tier = std::clamp(classification.materialTier, 1, 3);
	if (geometryRadius > 0.0f) {
		if (geometryRadius < settings.MinShallowWindowRadius)
			tier = std::min(tier, 1);
		else if (geometryRadius < settings.MinFullWindowRadius)
			tier = std::min(tier, 2);
	}

	data.Class0 = {
		static_cast<float>(tier),
		classification.diffuseNamedGlass ? 2.0f : (classification.namedGlass ? 1.0f : 0.0f),
		(classification.hasGlowTexture ? 1.0f : 0.0f) +
			(classification.hasAuthoredMask ? 2.0f : 0.0f),
		classification.diffuseNamedWindow ? 2.0f : (classification.explicitWindow ? 1.0f : 0.0f)
	};
	data.Asset0.x = static_cast<float>(classification.roomFamily);
	// Layout eligibility is a cached material property. 2 means an explicit
	// window with a native glow guide, 1 is a generic architectural-glass glow
	// guide, and 0 keeps the geometry fallback. Aperture bounds themselves still
	// require the current draw's UV/world derivatives and are resolved in HLSL.
	data.Layout0.y = classification.hasGlowTexture ? (classification.explicitWindow ? 2.0f : 1.0f) : (classification.hasAuthoredMask ? 1.0f : 0.0f);
	if (geometry) {
		const auto& center = sharedWindow.center;
		data.Geometry0 = { center.x, center.y, center.z, geometryRadius };
		data.Presentation1.w = sharedWindow.grouped ? 1.0f : 0.0f;

		// The NIF geometry name is stable across diffuse/normal texture replacers.
		// It lets dedicated aperture meshes use one auto-scaled room while large
		// walls/facades retain a repeating reference grid.
		const char* rawGeometryName = geometry->name.c_str();
		geometryNameScratch.assign(rawGeometryName ? rawGeometryName : "");
		std::transform(geometryNameScratch.begin(), geometryNameScratch.end(),
			geometryNameScratch.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		const std::string_view geometryName = geometryNameScratch;
		const bool dedicatedNameHint = ContainsAny(geometryName, { "window", "glass", "pane", "glazing" });
		const bool facadeNameHint = ContainsAny(geometryName, { "facade", "wall", "house", "building", "exterior" });

		const std::array<std::pair<float, float>, 6> familyRoomSizes{ { { 110.0f, 140.0f }, { 132.0f, 174.0f }, { 118.0f, 138.0f },
			{ 104.0f, 154.0f }, { 128.0f, 128.0f }, { 126.0f, 146.0f } } };
		const auto familyIndex = static_cast<std::size_t>(
			std::clamp(classification.roomFamily, 0, 5));
		const auto [familyWidth, familyHeight] = familyRoomSizes[familyIndex];
		const float referenceRadius = std::max(
			std::sqrt(familyWidth * familyWidth + familyHeight * familyHeight) * 0.5f,
			1.0f);
		const float radiusRatio = geometryRadius / referenceRadius;
		// A validated parent aperture represents one whole window even when
		// each child is an individually drawn triangle/pane mesh.
		float apertureHint = sharedWindow.grouped ? 1.0f : 0.0f;
		if (!sharedWindow.grouped && radiusRatio <= 2.75f)
			apertureHint = 1.0f;
		else if (!sharedWindow.grouped && radiusRatio >= 4.25f)
			apertureHint = -1.0f;
		else if (!sharedWindow.grouped && dedicatedNameHint != facadeNameHint)
			apertureHint = dedicatedNameHint ? 1.0f : -1.0f;

		// One mesh often carries several windows of a building. Centring a
		// single room on that mesh leaves every window showing a flat sliver
		// of it. When the material has a glow map or an installed glass mask,
		// classify such a mesh as a facade so the shader fits a room to each
		// window from that guide.
		const bool hasLayoutGuide = classification.hasGlowTexture || classification.hasAuthoredMask;
		const bool multiWindowMesh = settings.AutomaticRoomSizing && hasLayoutGuide &&
		                             !sharedWindow.grouped && radiusRatio > settings.MultiWindowMeshScale;
		if (multiWindowMesh)
			apertureHint = -1.0f;
		data.Asset0.w = apertureHint;

		if (settings.DebugWindowDetection && diag.geometryLines < kMaxGeometryLogLines) {
			const float centerDistance = center.GetDistance(Util::GetEyePosition(0));
			if (centerDistance - geometryRadius < 700.0f && diag.loggedGeometry.insert(geometry).second) {
				++diag.geometryLines;
				const char* parentName = geometry->parent && geometry->parent->name.c_str() ? geometry->parent->name.c_str() : "";
				logger::info(
					"[WindowLife] GEOMETRY '{}' parent='{}' radius={:.0f} ownRadius={:.0f} parentRadius={:.0f} grouped={} sizeRatio={:.2f} layout={} multiWindow={} guide={} tier={} cameraDistance={:.0f} texture='{}'",
					geometryNameScratch,
					parentName,
					geometryRadius,
					geometry->worldBound.radius,
					geometry->parent ? geometry->parent->worldBound.radius : 0.0f,
					sharedWindow.grouped ? 1 : 0,
					radiusRatio,
					apertureHint > 0.5f ? "single-window" : (apertureHint < -0.5f ? "per-window fit" : "undecided"),
					multiWindowMesh ? 1 : 0,
					classification.hasGlowTexture ? "glow" : (classification.hasAuthoredMask ? "mask" : "none"),
					tier,
					centerDistance,
					classification.evidence);
			}
		}

		// Grouped sibling panes must have an identical identity even if their
		// material names differ (e.g. centre glass versus stained-glass border).
		std::uint32_t instanceIdentity = sharedWindow.grouped ? sharedWindow.groupIdentity : classification.materialIdentity;
		if (!sharedWindow.grouped)
			HashCombine32(instanceIdentity, StableHash32(geometryName));
		// Eighth-unit quantization absorbs harmless transform noise while keeping
		// adjacent instances distinct even when they share the same NIF/material.
		HashCombine32(instanceIdentity, static_cast<std::uint32_t>(std::lround(center.x * 8.0f)));
		HashCombine32(instanceIdentity, static_cast<std::uint32_t>(std::lround(center.y * 8.0f)));
		HashCombine32(instanceIdentity, static_cast<std::uint32_t>(std::lround(center.z * 8.0f)));
		data.Layout0.x = static_cast<float>(instanceIdentity & 0x00FFFFFFu) / 16777216.0f;
	} else {
		data.Geometry0 = { 0.0f, 0.0f, 0.0f, geometryRadius };
		data.Layout0.x = static_cast<float>(classification.materialIdentity & 0x00FFFFFFu) / 16777216.0f;
	}

	if (!activeDataValid || std::memcmp(&data, &currentActiveData, sizeof(data)) != 0) {
		if (!UploadData(activeBuffer.get(), data)) {
			activeDataValid = false;
			BindNeutral();
			return;
		}
		currentActiveData = data;
		activeDataValid = true;
	}

	BindActive(classification);
}

void WindowLife::BSLightingShader_SetupGeometry(RE::BSRenderPass* pass)
{
	++diag.hookCalls;
	if (!diag.firstDrawLogged) {
		diag.firstDrawLogged = true;
		logger::info("[WindowLife] First Lighting draw reached the WindowLife geometry hook (enabled={}).", settings.EnableWindowLife);
	}
	if (settings.DebugWindowDetection)
		ReportDiagnostics();

	// The private structured SRV is rebound for EVERY Lighting draw so candidate state cannot leak.
	if (!pass || !pass->shaderProperty || !pass->shaderProperty->material) {
		++diag.noMaterial;
		BindNeutral();
		return;
	}
	if (!settings.EnableWindowLife) {
		++diag.disabled;
		BindNeutral();
		return;
	}

	// Malformed replacement meshes must never upload NaN bounds into the room
	// projector or overflow the stable instance-identity quantization.
	const auto* geometry = pass->geometry;
	if (!geometry || !std::isfinite(geometry->worldBound.radius) ||
		geometry->worldBound.radius < 0.0f || geometry->worldBound.radius > 1.0e8f ||
		!std::isfinite(geometry->worldBound.center.x) ||
		!std::isfinite(geometry->worldBound.center.y) ||
		!std::isfinite(geometry->worldBound.center.z) ||
		std::abs(geometry->worldBound.center.x) > 1.0e8f ||
		std::abs(geometry->worldBound.center.y) > 1.0e8f ||
		std::abs(geometry->worldBound.center.z) > 1.0e8f) {
		if (!invalidGeometryLogged) {
			logger::warn("[WindowLife] Lighting draw has missing or invalid geometry bounds; binding neutral WindowLife data.");
			invalidGeometryLogged = true;
		}
		++diag.invalidGeometry;
		BindNeutral();
		return;
	}

	auto* material = static_cast<RE::BSLightingShaderMaterialBase*>(pass->shaderProperty->material);
	const auto& classification = GetClassification(material);
	if (!classification.isWindow) {
		++diag.notWindow;
		BindNeutral();
		return;
	}

	++diag.windowDraws;
	UpdateAndBindActive(classification, geometry);
}

void WindowLife::ReportDiagnostics()
{
	const auto now = std::chrono::steady_clock::now();
	if (diag.lastReport.time_since_epoch().count() == 0) {
		diag.lastReport = now;
		return;
	}
	if (now - diag.lastReport < std::chrono::seconds(5))
		return;
	diag.lastReport = now;
	static constexpr const char* kDebugViewNames[] = {
		"Class Overlay", "Base Room Coordinates", "Final Room Coordinates",
		"Framed Room UV", "Room Identity", "Room Basis", "Pane Mask",
		"Room Box Surface", "Room Box Depth", "Glow Guide Signal", "Multi-pane Group"
	};
	logger::info(
		"[WindowLife] DIAG 5s: view='{}' threshold={:.2f} | draws={} noMaterial={} disabled={} invalidBounds={} notWindow={} windowDraws={} | new materials={} noTextureSet={} noTexturePaths={} cached={}",
		kDebugViewNames[std::clamp(settings.DebugRoomProjection, 0, 10)], settings.MultiWindowMeshScale,
		diag.hookCalls, diag.noMaterial, diag.disabled, diag.invalidGeometry, diag.notWindow, diag.windowDraws,
		diag.newMaterials, diag.noTextureSet, diag.noTexturePaths, classificationCache.size());
	diag.hookCalls = diag.noMaterial = diag.disabled = diag.invalidGeometry = diag.notWindow = diag.windowDraws = 0;
	diag.newMaterials = diag.noTextureSet = diag.noTexturePaths = 0;
}

void WindowLife::Hooks::BSLightingShader_SetupGeometry::thunk(
	RE::BSShader* a_this,
	RE::BSRenderPass* a_pass,
	std::uint32_t a_renderFlags)
{
	// Preserve all previously-installed per-draw hooks, then make WindowLife
	// structured per-draw SRV the final authoritative bind for this draw.
	func(a_this, a_pass, a_renderFlags);

	auto& windowLife = globals::features::windowLife;
	if (windowLife.loaded)
		windowLife.BSLightingShader_SetupGeometry(a_pass);
}

void WindowLife::PostPostLoad()
{
	logger::info("[WindowLife] Layered architectural-glass + inhabited-window renderer ACTIVE");
	Hooks::Install();
}

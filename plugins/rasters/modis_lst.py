"""MODIS land surface temperature (LST) streamed straight into OCS

The MODIS MOD11A1 (daily) and MOD11A2 (8-day) LST products are read as Cloud Optimized
GeoTIFFs through the Microsoft Planetary Computer STAC API. There is no account and no
manual download: for each period OCS asks for, the plugin searches the catalogue, reads
only the pixels that cover your instance extent, mosaics the MODIS tiles, reprojects them
from the MODIS sinusoidal grid to a regular lat/lon grid, converts to degrees Celsius, and
returns one ``(t, y, x)`` xarray.Dataset. OCS then stores, publishes and syncs it.

Cadences (``params.cadence`` in the template):
- ``monthly``: mean of the MOD11A2 8-day composites whose start date falls in the month.
- ``daily``:   MOD11A1.

``periods()`` asks the catalogue for the newest item that exists and stops there, so a
sync never invents periods the source has not published. A period *inside* the range that
has no items (for example a sensor outage) is stored as all-NaN with a warning, because
OCS needs a gap-free series.
"""

from __future__ import annotations

import asyncio
import logging
import math
import warnings
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr

from open_climate_service.streaming import (
    BaseDatasetPlugin,
    daily_period_ids,
    monthly_period_ids,
    normalize_period,
)

logger = logging.getLogger(__name__)

_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
_COLLECTIONS = {"monthly": "modis-11A2-061", "daily": "modis-11A1-061"}
_GRID_DEG = 1.0 / 120.0  # 30 arc-seconds, about 1 km: the MODIS Climate Modelling Grid resolution
_LST_SCALE = 0.02  # MOD11 digital number -> Kelvin
_KELVIN = 273.15
_COMPOSITE_DAYS = 8


class ModisLSTPlugin(BaseDatasetPlugin):
    """Streaming plugin: MODIS LST from the Planetary Computer STAC catalogue.

    Args (all from the template's ``ingestion.params``):
        cadence: ``monthly`` (MOD11A2) or ``daily`` (MOD11A1).
        band: ``LST_Day_1km`` or ``LST_Night_1km``.
        variable: name of the variable in the OCS store (must match the template).
        quality: ``any`` keeps every pixel MODIS produced a value for; ``good`` keeps only
            pixels whose QC flag says "good quality" (fewer pixels, cleaner values).
        collection: override the STAC collection id (default depends on cadence).
        stac_url: override the STAC API (default: Microsoft Planetary Computer).
        sign: sign asset URLs with the free Planetary Computer token (default true).
            Set false for a STAC API whose assets are already public.
    """

    max_concurrency = 4
    commit_batch_size = 12

    def __init__(
        self,
        cadence: str = "monthly",
        band: str = "LST_Day_1km",
        variable: str = "lst_day",
        quality: str = "any",
        collection: str | None = None,
        stac_url: str = _STAC_URL,
        sign: bool = True,
        **_: Any,
    ) -> None:
        if cadence not in _COLLECTIONS:
            raise ValueError(f"cadence must be 'monthly' or 'daily', got {cadence!r}")
        if band not in {"LST_Day_1km", "LST_Night_1km"}:
            raise ValueError(f"band must be 'LST_Day_1km' or 'LST_Night_1km', got {band!r}")
        if quality not in {"any", "good"}:
            raise ValueError(f"quality must be 'any' or 'good', got {quality!r}")
        self.cadence = cadence
        self.band = band
        self.qc_band = band.replace("LST_", "QC_").replace("_1km", "")  # LST_Day_1km -> QC_Day
        self.variable = variable
        self.quality = quality
        self.collection = collection or _COLLECTIONS[cadence]
        self.stac_url = stac_url
        self.sign = sign

    # ------------------------------------------------------------------ periods
    async def periods(self, start: str, end: str) -> list[str]:
        """Ordered period ids, clamped to the newest data the catalogue actually holds."""
        latest = await asyncio.to_thread(self._latest_item_day)
        if self.cadence == "monthly":
            # Composites are ordered, so once an item exists in month M the month before it is complete.
            return monthly_period_ids(start, end, cutoff=latest.replace(day=1) - timedelta(days=1))
        return daily_period_ids(start, end, cutoff=latest)

    # ------------------------------------------------------------------ fetch
    def fetch_period(self, period_id: str, bbox: list[float], **_: Any) -> xr.Dataset:
        """Read one period from the catalogue and return it as a one-step dataset."""
        import odc.stac
        import rioxarray  # noqa: F401  # activates the .rio accessor

        start, end, stamp = self._window(period_id)
        geobox = self._geobox(bbox)
        items = self._items(start, end, bbox)

        if not items:
            logger.warning(
                "No %s items for %s; storing an all-NaN period to keep the series gap-free",
                self.collection,
                period_id,
            )
            lst = self._all_nan(geobox)
        else:
            bands = [self.band] + ([self.qc_band] if self.quality == "good" else [])
            cube = odc.stac.load(items, bands=bands, geobox=geobox, resampling="nearest", groupby="time")
            raw = cube[self.band].astype("float32")
            valid = raw > 0  # 0 is the MOD11 fill value (NaN > 0 is False, so NaN stays masked)
            if self.quality == "good":
                valid = valid & ((cube[self.qc_band].astype("uint8") & 3) == 0)  # QC bits 0-1 == 00
            kelvin = raw.where(valid) * _LST_SCALE
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)  # all-cloud pixels average to NaN
                lst = (kelvin.mean("time", skipna=True) - _KELVIN).astype("float32")

        lst.attrs = {}
        lst.encoding = {}
        lst = lst.rio.write_crs("EPSG:4326")
        return normalize_period(lst, variable=self.variable, period=stamp, bbox=bbox)

    # ------------------------------------------------------------------ helpers
    def _catalog(self) -> Any:
        import pystac_client

        if self.sign:
            import planetary_computer

            return pystac_client.Client.open(self.stac_url, modifier=planetary_computer.sign_inplace)
        return pystac_client.Client.open(self.stac_url)

    def _latest_item_day(self) -> date:
        """Start date of the newest item in the collection."""
        search = self._catalog().search(collections=[self.collection], max_items=1, sortby="-datetime")
        for item in search.items():
            return self._item_day(item)
        raise RuntimeError(f"No items found in STAC collection {self.collection!r} at {self.stac_url}")

    def _items(self, start: date, end: date, bbox: list[float]) -> list[Any]:
        """Items whose start date lies in ``[start, end)`` and that touch the extent."""
        # Search a little wider than the window, then filter on the item's own start date, so an
        # 8-day composite is counted in exactly one month (the month it starts in).
        lookback = timedelta(days=_COMPOSITE_DAYS)
        search = self._catalog().search(
            collections=[self.collection],
            bbox=[float(v) for v in bbox],
            datetime=f"{(start - lookback).isoformat()}/{end.isoformat()}",
        )
        return [item for item in search.items() if start <= self._item_day(item) < end]

    @staticmethod
    def _item_day(item: Any) -> date:
        """The day an item's period starts (``datetime``, else ``start_datetime``)."""
        value = item.datetime or item.properties.get("start_datetime")
        if value is None:
            raise ValueError(f"STAC item {item.id!r} has neither datetime nor start_datetime")
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return value.date()

    def _window(self, period_id: str) -> tuple[date, date, str]:
        """Window ``[start, end)`` and the timestamp stored for the period."""
        if self.cadence == "monthly":
            year, month = int(period_id[:4]), int(period_id[5:7])
            start = date(year, month, 1)
            end = date(year + (month == 12), month % 12 + 1, 1)
            return start, end, start.isoformat()
        day = date.fromisoformat(period_id)
        return day, day + timedelta(days=1), day.isoformat()

    @staticmethod
    def _grid(bbox: list[float]) -> tuple[float, float, int, int]:
        """Snap the extent to a fixed 30-arc-second grid so every period shares one grid."""
        xmin, ymin, xmax, ymax = (float(v) for v in bbox)
        g = _GRID_DEG
        x0 = math.floor(xmin / g) * g
        y1 = math.ceil(ymax / g) * g
        width = max(1, math.ceil((xmax - x0) / g - 1e-9))
        height = max(1, math.ceil((y1 - ymin) / g - 1e-9))
        return x0, y1, width, height

    def _geobox(self, bbox: list[float]) -> Any:
        from odc.geo.geobox import GeoBox

        x0, y1, width, height = self._grid(bbox)
        g = _GRID_DEG
        return GeoBox.from_bbox((x0, y1 - height * g, x0 + width * g, y1), crs="EPSG:4326", shape=(height, width))

    @staticmethod
    def _all_nan(geobox: Any) -> xr.DataArray:
        from odc.geo.xr import xr_zeros

        return (xr_zeros(geobox, dtype="float32") * np.nan).astype("float32")

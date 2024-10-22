//! NOTE: this file is autogenerated, DO NOT MODIFY
//--------------------------------------------------------------------------------
// Section: Constants (1)
//--------------------------------------------------------------------------------
pub const E_UNKNOWNTYPE = @import("../../zig.zig").typedConst(HRESULT, @as(i32, -2144665560));

//--------------------------------------------------------------------------------
// Section: Types (21)
//--------------------------------------------------------------------------------
pub const VisualMutationType = enum(i32) {
    Add = 0,
    Remove = 1,
};
pub const Add = VisualMutationType.Add;
pub const Remove = VisualMutationType.Remove;

pub const BaseValueSource = enum(i32) {
    BaseValueSourceUnknown = 0,
    BaseValueSourceDefault = 1,
    BaseValueSourceBuiltInStyle = 2,
    BaseValueSourceStyle = 3,
    BaseValueSourceLocal = 4,
    Inherited = 5,
    DefaultStyleTrigger = 6,
    TemplateTrigger = 7,
    StyleTrigger = 8,
    ImplicitStyleReference = 9,
    ParentTemplate = 10,
    ParentTemplateTrigger = 11,
    Animation = 12,
    Coercion = 13,
    BaseValueSourceVisualState = 14,
};
pub const BaseValueSourceUnknown = BaseValueSource.BaseValueSourceUnknown;
pub const BaseValueSourceDefault = BaseValueSource.BaseValueSourceDefault;
pub const BaseValueSourceBuiltInStyle = BaseValueSource.BaseValueSourceBuiltInStyle;
pub const BaseValueSourceStyle = BaseValueSource.BaseValueSourceStyle;
pub const BaseValueSourceLocal = BaseValueSource.BaseValueSourceLocal;
pub const Inherited = BaseValueSource.Inherited;
pub const DefaultStyleTrigger = BaseValueSource.DefaultStyleTrigger;
pub const TemplateTrigger = BaseValueSource.TemplateTrigger;
pub const StyleTrigger = BaseValueSource.StyleTrigger;
pub const ImplicitStyleReference = BaseValueSource.ImplicitStyleReference;
pub const ParentTemplate = BaseValueSource.ParentTemplate;
pub const ParentTemplateTrigger = BaseValueSource.ParentTemplateTrigger;
pub const Animation = BaseValueSource.Animation;
pub const Coercion = BaseValueSource.Coercion;
pub const BaseValueSourceVisualState = BaseValueSource.BaseValueSourceVisualState;

pub const SourceInfo = extern struct {
    FileName: ?BSTR,
    LineNumber: u32,
    ColumnNumber: u32,
    CharPosition: u32,
    Hash: ?BSTR,
};

pub const ParentChildRelation = extern struct {
    Parent: u64,
    Child: u64,
    ChildIndex: u32,
};

pub const VisualElement = extern struct {
    Handle: u64,
    SrcInfo: SourceInfo,
    Type: ?BSTR,
    Name: ?BSTR,
    NumChildren: u32,
};

pub const PropertyChainSource = extern struct {
    Handle: u64,
    TargetType: ?BSTR,
    Name: ?BSTR,
    Source: BaseValueSource,
    SrcInfo: SourceInfo,
};

pub const MetadataBit = enum(i32) {
    None = 0,
    ValueHandle = 1,
    PropertyReadOnly = 2,
    ValueCollection = 4,
    ValueCollectionReadOnly = 8,
    ValueBindingExpression = 16,
    ValueNull = 32,
    ValueHandleAndEvaluatedValue = 64,
};
// NOTE: not creating aliases because this enum is 'Scoped'

pub const PropertyChainValue = extern struct {
    Index: u32,
    Type: ?BSTR,
    DeclaringType: ?BSTR,
    ValueType: ?BSTR,
    ItemType: ?BSTR,
    Value: ?BSTR,
    Overridden: BOOL,
    MetadataBits: i64,
    PropertyName: ?BSTR,
    PropertyChainIndex: u32,
};

pub const EnumType = extern struct {
    Name: ?BSTR,
    ValueInts: ?*SAFEARRAY,
    ValueStrings: ?*SAFEARRAY,
};

pub const CollectionElementValue = extern struct {
    Index: u32,
    ValueType: ?BSTR,
    Value: ?BSTR,
    MetadataBits: i64,
};

pub const RenderTargetBitmapOptions = enum(i32) {
    t = 0,
    AndChildren = 1,
};
pub const RenderTarget = RenderTargetBitmapOptions.t;
pub const RenderTargetAndChildren = RenderTargetBitmapOptions.AndChildren;

pub const BitmapDescription = extern struct {
    Width: u32,
    Height: u32,
    Format: DXGI_FORMAT,
    AlphaMode: DXGI_ALPHA_MODE,
};

pub const ResourceType = enum(i32) {
    Static = 0,
    Theme = 1,
};
pub const ResourceTypeStatic = ResourceType.Static;
pub const ResourceTypeTheme = ResourceType.Theme;

pub const VisualElementState = enum(i32) {
    Resolved = 0,
    ResourceNotFound = 1,
    InvalidResource = 2,
};
pub const ErrorResolved = VisualElementState.Resolved;
pub const ErrorResourceNotFound = VisualElementState.ResourceNotFound;
pub const ErrorInvalidResource = VisualElementState.InvalidResource;

// TODO: this type is limited to platform 'windows10.0.10240'
const IID_IVisualTreeServiceCallback_Value = Guid.initString("aa7a8931-80e4-4fec-8f3b-553f87b4966e");
pub const IID_IVisualTreeServiceCallback = &IID_IVisualTreeServiceCallback_Value;
pub const IVisualTreeServiceCallback = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        OnVisualTreeChange: *const fn (
            self: *const IVisualTreeServiceCallback,
            relation: ParentChildRelation,
            element: VisualElement,
            mutation_type: VisualMutationType,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn onVisualTreeChange(self: *const T, relation_: ParentChildRelation, element_: VisualElement, mutation_type_: VisualMutationType) HRESULT {
                return @as(*const IVisualTreeServiceCallback.VTable, @ptrCast(self.vtable)).OnVisualTreeChange(@as(*const IVisualTreeServiceCallback, @ptrCast(self)), relation_, element_, mutation_type_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows10.0.14393'
const IID_IVisualTreeServiceCallback2_Value = Guid.initString("bad9eb88-ae77-4397-b948-5fa2db0a19ea");
pub const IID_IVisualTreeServiceCallback2 = &IID_IVisualTreeServiceCallback2_Value;
pub const IVisualTreeServiceCallback2 = extern struct {
    pub const VTable = extern struct {
        base: IVisualTreeServiceCallback.VTable,
        OnElementStateChanged: *const fn (
            self: *const IVisualTreeServiceCallback2,
            element: u64,
            element_state: VisualElementState,
            context: ?[*:0]const u16,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IVisualTreeServiceCallback.MethodMixin(T);
            pub inline fn onElementStateChanged(self: *const T, element_: u64, element_state_: VisualElementState, context_: ?[*:0]const u16) HRESULT {
                return @as(*const IVisualTreeServiceCallback2.VTable, @ptrCast(self.vtable)).OnElementStateChanged(@as(*const IVisualTreeServiceCallback2, @ptrCast(self)), element_, element_state_, context_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

const IID_IVisualTreeService_Value = Guid.initString("a593b11a-d17f-48bb-8f66-83910731c8a5");
pub const IID_IVisualTreeService = &IID_IVisualTreeService_Value;
pub const IVisualTreeService = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        AdviseVisualTreeChange: *const fn (
            self: *const IVisualTreeService,
            p_callback: ?*IVisualTreeServiceCallback,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        UnadviseVisualTreeChange: *const fn (
            self: *const IVisualTreeService,
            p_callback: ?*IVisualTreeServiceCallback,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetEnums: *const fn (
            self: *const IVisualTreeService,
            p_count: ?*u32,
            pp_enums: [*]?*EnumType,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        CreateInstance: *const fn (
            self: *const IVisualTreeService,
            type_name: ?BSTR,
            value: ?BSTR,
            p_instance_handle: ?*u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetPropertyValuesChain: *const fn (
            self: *const IVisualTreeService,
            instance_handle: u64,
            p_source_count: ?*u32,
            pp_property_sources: [*]?*PropertyChainSource,
            p_property_count: ?*u32,
            pp_property_values: [*]?*PropertyChainValue,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        SetProperty: *const fn (
            self: *const IVisualTreeService,
            instance_handle: u64,
            value: u64,
            property_index: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ClearProperty: *const fn (
            self: *const IVisualTreeService,
            instance_handle: u64,
            property_index: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetCollectionCount: *const fn (
            self: *const IVisualTreeService,
            instance_handle: u64,
            p_collection_size: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetCollectionElements: *const fn (
            self: *const IVisualTreeService,
            instance_handle: u64,
            start_index: u32,
            p_element_count: ?*u32,
            pp_element_values: [*]?*CollectionElementValue,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        AddChild: *const fn (
            self: *const IVisualTreeService,
            parent: u64,
            child: u64,
            index: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RemoveChild: *const fn (
            self: *const IVisualTreeService,
            parent: u64,
            index: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ClearChildren: *const fn (
            self: *const IVisualTreeService,
            parent: u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn adviseVisualTreeChange(self: *const T, p_callback_: ?*IVisualTreeServiceCallback) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).AdviseVisualTreeChange(@as(*const IVisualTreeService, @ptrCast(self)), p_callback_);
            }
            pub inline fn unadviseVisualTreeChange(self: *const T, p_callback_: ?*IVisualTreeServiceCallback) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).UnadviseVisualTreeChange(@as(*const IVisualTreeService, @ptrCast(self)), p_callback_);
            }
            pub inline fn getEnums(self: *const T, p_count_: ?*u32, pp_enums_: [*]?*EnumType) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).GetEnums(@as(*const IVisualTreeService, @ptrCast(self)), p_count_, pp_enums_);
            }
            pub inline fn createInstance(self: *const T, type_name_: ?BSTR, value_: ?BSTR, p_instance_handle_: ?*u64) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).CreateInstance(@as(*const IVisualTreeService, @ptrCast(self)), type_name_, value_, p_instance_handle_);
            }
            pub inline fn getPropertyValuesChain(self: *const T, instance_handle_: u64, p_source_count_: ?*u32, pp_property_sources_: [*]?*PropertyChainSource, p_property_count_: ?*u32, pp_property_values_: [*]?*PropertyChainValue) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).GetPropertyValuesChain(@as(*const IVisualTreeService, @ptrCast(self)), instance_handle_, p_source_count_, pp_property_sources_, p_property_count_, pp_property_values_);
            }
            pub inline fn setProperty(self: *const T, instance_handle_: u64, value_: u64, property_index_: u32) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).SetProperty(@as(*const IVisualTreeService, @ptrCast(self)), instance_handle_, value_, property_index_);
            }
            pub inline fn clearProperty(self: *const T, instance_handle_: u64, property_index_: u32) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).ClearProperty(@as(*const IVisualTreeService, @ptrCast(self)), instance_handle_, property_index_);
            }
            pub inline fn getCollectionCount(self: *const T, instance_handle_: u64, p_collection_size_: ?*u32) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).GetCollectionCount(@as(*const IVisualTreeService, @ptrCast(self)), instance_handle_, p_collection_size_);
            }
            pub inline fn getCollectionElements(self: *const T, instance_handle_: u64, start_index_: u32, p_element_count_: ?*u32, pp_element_values_: [*]?*CollectionElementValue) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).GetCollectionElements(@as(*const IVisualTreeService, @ptrCast(self)), instance_handle_, start_index_, p_element_count_, pp_element_values_);
            }
            pub inline fn addChild(self: *const T, parent_: u64, child_: u64, index_: u32) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).AddChild(@as(*const IVisualTreeService, @ptrCast(self)), parent_, child_, index_);
            }
            pub inline fn removeChild(self: *const T, parent_: u64, index_: u32) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).RemoveChild(@as(*const IVisualTreeService, @ptrCast(self)), parent_, index_);
            }
            pub inline fn clearChildren(self: *const T, parent_: u64) HRESULT {
                return @as(*const IVisualTreeService.VTable, @ptrCast(self.vtable)).ClearChildren(@as(*const IVisualTreeService, @ptrCast(self)), parent_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows10.0.10240'
const IID_IXamlDiagnostics_Value = Guid.initString("18c9e2b6-3f43-4116-9f2b-ff935d7770d2");
pub const IID_IXamlDiagnostics = &IID_IXamlDiagnostics_Value;
pub const IXamlDiagnostics = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        GetDispatcher: *const fn (
            self: *const IXamlDiagnostics,
            pp_dispatcher: ?*?*IInspectable,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetUiLayer: *const fn (
            self: *const IXamlDiagnostics,
            pp_layer: ?*?*IInspectable,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetApplication: *const fn (
            self: *const IXamlDiagnostics,
            pp_application: ?*?*IInspectable,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetIInspectableFromHandle: *const fn (
            self: *const IXamlDiagnostics,
            instance_handle: u64,
            pp_instance: ?*?*IInspectable,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetHandleFromIInspectable: *const fn (
            self: *const IXamlDiagnostics,
            p_instance: ?*IInspectable,
            p_handle: ?*u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        HitTest: *const fn (
            self: *const IXamlDiagnostics,
            rect: RECT,
            p_count: ?*u32,
            pp_instance_handles: [*]?*u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RegisterInstance: *const fn (
            self: *const IXamlDiagnostics,
            p_instance: ?*IInspectable,
            p_instance_handle: ?*u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetInitializationData: *const fn (
            self: *const IXamlDiagnostics,
            p_initialization_data: ?*?BSTR,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn getDispatcher(self: *const T, pp_dispatcher_: ?*?*IInspectable) HRESULT {
                return @as(*const IXamlDiagnostics.VTable, @ptrCast(self.vtable)).GetDispatcher(@as(*const IXamlDiagnostics, @ptrCast(self)), pp_dispatcher_);
            }
            pub inline fn getUiLayer(self: *const T, pp_layer_: ?*?*IInspectable) HRESULT {
                return @as(*const IXamlDiagnostics.VTable, @ptrCast(self.vtable)).GetUiLayer(@as(*const IXamlDiagnostics, @ptrCast(self)), pp_layer_);
            }
            pub inline fn getApplication(self: *const T, pp_application_: ?*?*IInspectable) HRESULT {
                return @as(*const IXamlDiagnostics.VTable, @ptrCast(self.vtable)).GetApplication(@as(*const IXamlDiagnostics, @ptrCast(self)), pp_application_);
            }
            pub inline fn getIInspectableFromHandle(self: *const T, instance_handle_: u64, pp_instance_: ?*?*IInspectable) HRESULT {
                return @as(*const IXamlDiagnostics.VTable, @ptrCast(self.vtable)).GetIInspectableFromHandle(@as(*const IXamlDiagnostics, @ptrCast(self)), instance_handle_, pp_instance_);
            }
            pub inline fn getHandleFromIInspectable(self: *const T, p_instance_: ?*IInspectable, p_handle_: ?*u64) HRESULT {
                return @as(*const IXamlDiagnostics.VTable, @ptrCast(self.vtable)).GetHandleFromIInspectable(@as(*const IXamlDiagnostics, @ptrCast(self)), p_instance_, p_handle_);
            }
            pub inline fn hitTest(self: *const T, rect_: RECT, p_count_: ?*u32, pp_instance_handles_: [*]?*u64) HRESULT {
                return @as(*const IXamlDiagnostics.VTable, @ptrCast(self.vtable)).HitTest(@as(*const IXamlDiagnostics, @ptrCast(self)), rect_, p_count_, pp_instance_handles_);
            }
            pub inline fn registerInstance(self: *const T, p_instance_: ?*IInspectable, p_instance_handle_: ?*u64) HRESULT {
                return @as(*const IXamlDiagnostics.VTable, @ptrCast(self.vtable)).RegisterInstance(@as(*const IXamlDiagnostics, @ptrCast(self)), p_instance_, p_instance_handle_);
            }
            pub inline fn getInitializationData(self: *const T, p_initialization_data_: ?*?BSTR) HRESULT {
                return @as(*const IXamlDiagnostics.VTable, @ptrCast(self.vtable)).GetInitializationData(@as(*const IXamlDiagnostics, @ptrCast(self)), p_initialization_data_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows10.0.14393'
const IID_IBitmapData_Value = Guid.initString("d1a34ef2-cad8-4635-a3d2-fcda8d3f3caf");
pub const IID_IBitmapData = &IID_IBitmapData_Value;
pub const IBitmapData = extern struct {
    pub const VTable = extern struct {
        base: IUnknown.VTable,
        CopyBytesTo: *const fn (
            self: *const IBitmapData,
            source_offset_in_bytes: u32,
            max_bytes_to_copy: u32,
            pv_bytes: [*:0]u8,
            number_of_bytes_copied: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetStride: *const fn (
            self: *const IBitmapData,
            p_stride: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetBitmapDescription: *const fn (
            self: *const IBitmapData,
            p_bitmap_description: ?*BitmapDescription,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetSourceBitmapDescription: *const fn (
            self: *const IBitmapData,
            p_bitmap_description: ?*BitmapDescription,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IUnknown.MethodMixin(T);
            pub inline fn copyBytesTo(self: *const T, source_offset_in_bytes_: u32, max_bytes_to_copy_: u32, pv_bytes_: [*:0]u8, number_of_bytes_copied_: ?*u32) HRESULT {
                return @as(*const IBitmapData.VTable, @ptrCast(self.vtable)).CopyBytesTo(@as(*const IBitmapData, @ptrCast(self)), source_offset_in_bytes_, max_bytes_to_copy_, pv_bytes_, number_of_bytes_copied_);
            }
            pub inline fn getStride(self: *const T, p_stride_: ?*u32) HRESULT {
                return @as(*const IBitmapData.VTable, @ptrCast(self.vtable)).GetStride(@as(*const IBitmapData, @ptrCast(self)), p_stride_);
            }
            pub inline fn getBitmapDescription(self: *const T, p_bitmap_description_: ?*BitmapDescription) HRESULT {
                return @as(*const IBitmapData.VTable, @ptrCast(self.vtable)).GetBitmapDescription(@as(*const IBitmapData, @ptrCast(self)), p_bitmap_description_);
            }
            pub inline fn getSourceBitmapDescription(self: *const T, p_bitmap_description_: ?*BitmapDescription) HRESULT {
                return @as(*const IBitmapData.VTable, @ptrCast(self.vtable)).GetSourceBitmapDescription(@as(*const IBitmapData, @ptrCast(self)), p_bitmap_description_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows10.0.14393'
const IID_IVisualTreeService2_Value = Guid.initString("130f5136-ec43-4f61-89c7-9801a36d2e95");
pub const IID_IVisualTreeService2 = &IID_IVisualTreeService2_Value;
pub const IVisualTreeService2 = extern struct {
    pub const VTable = extern struct {
        base: IVisualTreeService.VTable,
        GetPropertyIndex: *const fn (
            self: *const IVisualTreeService2,
            object: u64,
            property_name: ?[*:0]const u16,
            p_property_index: ?*u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetProperty: *const fn (
            self: *const IVisualTreeService2,
            object: u64,
            property_index: u32,
            p_value: ?*u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        ReplaceResource: *const fn (
            self: *const IVisualTreeService2,
            resource_dictionary: u64,
            key: u64,
            new_value: u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RenderTargetBitmap: *const fn (
            self: *const IVisualTreeService2,
            handle: u64,
            options: RenderTargetBitmapOptions,
            max_pixel_width: u32,
            max_pixel_height: u32,
            pp_bitmap_data: ?*?*IBitmapData,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IVisualTreeService.MethodMixin(T);
            pub inline fn getPropertyIndex(self: *const T, object_: u64, property_name_: ?[*:0]const u16, p_property_index_: ?*u32) HRESULT {
                return @as(*const IVisualTreeService2.VTable, @ptrCast(self.vtable)).GetPropertyIndex(@as(*const IVisualTreeService2, @ptrCast(self)), object_, property_name_, p_property_index_);
            }
            pub inline fn getProperty(self: *const T, object_: u64, property_index_: u32, p_value_: ?*u64) HRESULT {
                return @as(*const IVisualTreeService2.VTable, @ptrCast(self.vtable)).GetProperty(@as(*const IVisualTreeService2, @ptrCast(self)), object_, property_index_, p_value_);
            }
            pub inline fn replaceResource(self: *const T, resource_dictionary_: u64, key_: u64, new_value_: u64) HRESULT {
                return @as(*const IVisualTreeService2.VTable, @ptrCast(self.vtable)).ReplaceResource(@as(*const IVisualTreeService2, @ptrCast(self)), resource_dictionary_, key_, new_value_);
            }
            pub inline fn renderTargetBitmap(self: *const T, handle_: u64, options_: RenderTargetBitmapOptions, max_pixel_width_: u32, max_pixel_height_: u32, pp_bitmap_data_: ?*?*IBitmapData) HRESULT {
                return @as(*const IVisualTreeService2.VTable, @ptrCast(self.vtable)).RenderTargetBitmap(@as(*const IVisualTreeService2, @ptrCast(self)), handle_, options_, max_pixel_width_, max_pixel_height_, pp_bitmap_data_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

// TODO: this type is limited to platform 'windows10.0.15063'
const IID_IVisualTreeService3_Value = Guid.initString("0e79c6e0-85a0-4be8-b41a-655cf1fd19bd");
pub const IID_IVisualTreeService3 = &IID_IVisualTreeService3_Value;
pub const IVisualTreeService3 = extern struct {
    pub const VTable = extern struct {
        base: IVisualTreeService2.VTable,
        ResolveResource: *const fn (
            self: *const IVisualTreeService3,
            resource_context: u64,
            resource_name: ?[*:0]const u16,
            resource_type: ResourceType,
            property_index: u32,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        GetDictionaryItem: *const fn (
            self: *const IVisualTreeService3,
            dictionary_handle: u64,
            resource_name: ?[*:0]const u16,
            resource_is_implicit_style: BOOL,
            resource_handle: ?*u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        AddDictionaryItem: *const fn (
            self: *const IVisualTreeService3,
            dictionary_handle: u64,
            resource_key: u64,
            resource_handle: u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
        RemoveDictionaryItem: *const fn (
            self: *const IVisualTreeService3,
            dictionary_handle: u64,
            resource_key: u64,
        ) callconv(@import("std").os.windows.WINAPI) HRESULT,
    };
    vtable: *const VTable,
    pub fn MethodMixin(comptime T: type) type {
        return struct {
            pub usingnamespace IVisualTreeService2.MethodMixin(T);
            pub inline fn resolveResource(self: *const T, resource_context_: u64, resource_name_: ?[*:0]const u16, resource_type_: ResourceType, property_index_: u32) HRESULT {
                return @as(*const IVisualTreeService3.VTable, @ptrCast(self.vtable)).ResolveResource(@as(*const IVisualTreeService3, @ptrCast(self)), resource_context_, resource_name_, resource_type_, property_index_);
            }
            pub inline fn getDictionaryItem(self: *const T, dictionary_handle_: u64, resource_name_: ?[*:0]const u16, resource_is_implicit_style_: BOOL, resource_handle_: ?*u64) HRESULT {
                return @as(*const IVisualTreeService3.VTable, @ptrCast(self.vtable)).GetDictionaryItem(@as(*const IVisualTreeService3, @ptrCast(self)), dictionary_handle_, resource_name_, resource_is_implicit_style_, resource_handle_);
            }
            pub inline fn addDictionaryItem(self: *const T, dictionary_handle_: u64, resource_key_: u64, resource_handle_: u64) HRESULT {
                return @as(*const IVisualTreeService3.VTable, @ptrCast(self.vtable)).AddDictionaryItem(@as(*const IVisualTreeService3, @ptrCast(self)), dictionary_handle_, resource_key_, resource_handle_);
            }
            pub inline fn removeDictionaryItem(self: *const T, dictionary_handle_: u64, resource_key_: u64) HRESULT {
                return @as(*const IVisualTreeService3.VTable, @ptrCast(self.vtable)).RemoveDictionaryItem(@as(*const IVisualTreeService3, @ptrCast(self)), dictionary_handle_, resource_key_);
            }
        };
    }
    pub usingnamespace MethodMixin(@This());
};

//--------------------------------------------------------------------------------
// Section: Functions (2)
//--------------------------------------------------------------------------------
pub extern "windows.ui.xaml" fn InitializeXamlDiagnostic(
    end_point_name: ?[*:0]const u16,
    pid: u32,
    wsz_dll_xaml_diagnostics: ?[*:0]const u16,
    wsz_t_a_p_dll_name: ?[*:0]const u16,
    tap_clsid: Guid,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

// TODO: this type is limited to platform 'windows10.0.15063'
pub extern "windows.ui.xaml" fn InitializeXamlDiagnosticsEx(
    end_point_name: ?[*:0]const u16,
    pid: u32,
    wsz_dll_xaml_diagnostics: ?[*:0]const u16,
    wsz_t_a_p_dll_name: ?[*:0]const u16,
    tap_clsid: Guid,
    wsz_initialization_data: ?[*:0]const u16,
) callconv(@import("std").os.windows.WINAPI) HRESULT;

//--------------------------------------------------------------------------------
// Section: Unicode Aliases (0)
//--------------------------------------------------------------------------------
const thismodule = @This();
pub usingnamespace switch (@import("../../zig.zig").unicode_mode) {
    .ansi => struct {},
    .wide => struct {},
    .unspecified => if (@import("builtin").is_test) struct {} else struct {},
};
//--------------------------------------------------------------------------------
// Section: Imports (11)
//--------------------------------------------------------------------------------
const Guid = @import("../../zig.zig").Guid;
const BOOL = @import("../../foundation.zig").BOOL;
const BSTR = @import("../../foundation.zig").BSTR;
const DXGI_ALPHA_MODE = @import("../../graphics/dxgi/common.zig").DXGI_ALPHA_MODE;
const DXGI_FORMAT = @import("../../graphics/dxgi/common.zig").DXGI_FORMAT;
const HRESULT = @import("../../foundation.zig").HRESULT;
const IInspectable = @import("../../system/win_rt.zig").IInspectable;
const IUnknown = @import("../../system/com.zig").IUnknown;
const PWSTR = @import("../../foundation.zig").PWSTR;
const RECT = @import("../../foundation.zig").RECT;
const SAFEARRAY = @import("../../system/com.zig").SAFEARRAY;

test {
    @setEvalBranchQuota(comptime @import("std").meta.declarations(@This()).len * 3);

    // reference all the pub declarations
    if (!@import("builtin").is_test) return;
    inline for (comptime @import("std").meta.declarations(@This())) |decl| {
        _ = @field(@This(), decl.name);
    }
}
